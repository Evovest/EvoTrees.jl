using KernelAbstractions
using Atomix

# Update observation-to-node assignments (left=node*2, right=node*2+1)
@kernel function update_nodes_idx_kernel!(
	nidx::AbstractVector{T},        # Node index for each observation (in/out)
	@Const(is),                     # Observation indices to process
	@Const(x_bin),                  # Binned feature values [n_obs, n_feats]
	@Const(cond_feats),             # Split feature for each node
	@Const(cond_bins),              # Split threshold for each node
	@Const(feattypes),              # Feature types (true=numeric, false=categorical)
) where {T <: Unsigned}
	gidx = @index(Global)
	@inbounds if gidx <= length(is)
		obs = is[gidx]              # Get observation index
		node = nidx[obs]            # Get current node for this observation
		if node > 0                 # If observation is in an active node
			feat = cond_feats[node] # Get split feature for this node
			bin = cond_bins[node]   # Get split threshold
			# If bin == 0, node is a leaf - keep the current node ID
			if bin != 0
				feattype = feattypes[feat]
				is_left = feattype ? (x_bin[obs, feat] <= bin) : (x_bin[obs, feat] == bin)
				nidx[obs] = (node << 1) + T(Int(!is_left))
			end
			# If bin == 0, do nothing - nidx[obs] already has the leaf node ID
		end
	end
end

# Build gradient histograms: h∇[2K+1, n_bins, n_feats, n_nodes]
@kernel function hist_kernel!(
	h∇::AbstractArray{T, 4},         # Histogram [2K+1, n_bins, n_feats, n_nodes]
	@Const(∇),                      # Gradients [2K+1, n_obs]
	@Const(x_bin),                  # Binned features [n_obs, n_feats]
	@Const(nidx),                   # Node index for each observation
	@Const(js),                     # Feature indices to process
	@Const(is),                     # Observation indices to process
	K::Int,                         # Number of output dimensions
	chunk_size::Int,                # Observations per thread (reduces contention)
	@Const(target_mask)             # Mask indicating which nodes to build histograms for
) where {T}
	gidx = @index(Global, Linear)

	n_feats = length(js)
	n_obs = length(is)
	total_chunks = cld(n_obs, chunk_size)
	total_threads = n_feats * total_chunks

	if gidx <= total_threads
		feat_idx = (gidx - 1) % n_feats + 1
		chunk_idx = (gidx - 1) ÷ n_feats
		feat = js[feat_idx]

		start_obs = chunk_idx * chunk_size + 1
		end_obs = min(start_obs + chunk_size - 1, n_obs)

		@inbounds for obs_idx in start_obs:end_obs
			obs = is[obs_idx]
			node = nidx[obs]

			if node > 0 && node <= size(h∇, 4) && target_mask[node] != 0
				bin = x_bin[obs, feat]

				if bin > 0 && bin <= size(h∇, 2)
					for k in 1:(2*K+1)
						grad = ∇[k, obs]
						Atomix.@atomic h∇[k, bin, feat, node] += grad
					end
				end
			end
		end
	end
end

# Split active siblings into BUILD (smaller) vs SUBTRACT (larger) lists; sibling via node ⊻ 1
@kernel function separate_nodes_kernel!(
	build_nodes, build_count,       # Output: nodes to build via observation scan
	subtract_nodes, subtract_count, # Output: nodes to compute via subtraction
	@Const(active_nodes),           # Input: all active child nodes at current depth
	@Const(node_counts)             # Input: raw counts per node (number of observations)
)
	idx = @index(Global)
	@inbounds if idx <= length(active_nodes)
		node = active_nodes[idx]

		if node > 0
			sibling = node ⊻ 1

			# Compare raw observation counts (not weights)
			w_node = node_counts[node]
			w_sibling = node_counts[sibling]

			# Tiebreak by node id on equality
			if w_node < w_sibling || (w_node == w_sibling && node < sibling)
				pos = Atomix.@atomic build_count[1] += 1
				build_nodes[pos] = node
			else
				pos = Atomix.@atomic subtract_count[1] += 1
				subtract_nodes[pos] = node
			end
		end
	end
end

# Compute hist via subtraction: h∇[child] = h∇[parent] - h∇[sibling]
@kernel function subtract_hist_kernel!(
	h∇,                    # Histogram [2K+1, n_bins, n_feats, n_nodes] - modified in-place
	@Const(subtract_nodes) # List of larger children to compute via subtraction
)
	gidx = @index(Global)

	# Decode histogram dimensions to parallelize across all elements
	n_k = size(h∇, 1)
	n_b = size(h∇, 2)
	n_j = size(h∇, 3)
	n_elements_per_node = n_k * n_b * n_j

	node_idx = (gidx - 1) ÷ n_elements_per_node + 1

	if node_idx <= length(subtract_nodes)
		remainder = (gidx - 1) % n_elements_per_node
		j = remainder ÷ (n_k * n_b) + 1
		remainder = remainder % (n_k * n_b)
		b = remainder ÷ n_k + 1
		k = remainder % n_k + 1

		@inbounds node = subtract_nodes[node_idx]

		if node > 0
			parent = node >> 1
			sibling = node ⊻ 1

			@inbounds h∇[k, b, j, node] = h∇[k, b, j, parent] - h∇[k, b, j, sibling]
		end
	end
end

# Accumulate gradients for root node
@kernel function reduce_root_sums_kernel!(nodes_sum, @Const(∇), @Const(is))
	idx = @index(Global)
	if idx <= length(is)
		obs = is[idx]
		n_k = size(∇, 1)
		@inbounds for k in 1:n_k
			Atomix.@atomic nodes_sum[k, 1] += ∇[k, obs]
		end
	end
end

# Find best split per node from histograms (supports multiple losses/constraints)
@kernel function find_best_split_from_hist_kernel!(
	::Type{L},                      # Loss function type
	gains::AbstractVector{T},       # Output: best gain for each node
	bins::AbstractVector{Int32},    # Output: best bin for each node
	feats::AbstractVector{Int32},   # Output: best feature for each node
	@Const(h∇),                     # Input: histograms [2K+1, n_bins, n_feats, n_nodes]
	nodes_sum,                      # Input/Output: gradient sums per node
	@Const(active_nodes),           # Input: which nodes to process
	@Const(js),                     # Input: feature indices to consider
	@Const(feattypes),              # Input: feature types (numeric/categorical)
	@Const(monotone_constraints),   # Input: monotonicity constraints per feature
	lambda::T,                      # Regularization: L1-like penalty
	L2::T,                          # Regularization: L2 penalty
	min_weight::T,                  # Minimum observations per leaf
	K::Int,                         # Number of output dimensions
	sums_temp::AbstractArray{T, 2},   # Temporary storage for gradient accumulation
) where {T, L}
	n_idx = @index(Global)

	@inbounds if n_idx <= length(active_nodes)
		node = active_nodes[n_idx]
		if node == 0
			gains[n_idx] = T(-Inf)
			bins[n_idx] = Int32(0)
			feats[n_idx] = Int32(0)
		else
			nbins = size(h∇, 2)
			eps = T(1e-8)

			# Compute node total gradients by summing across bins
			if !isempty(js)
				first_feat = js[1]
				for k in 1:(2*K+1)
					sum_val = zero(T)
					for b in 1:nbins
						sum_val += h∇[k, b, first_feat, node]
					end
					nodes_sum[k, node] = sum_val
				end
			end

			w_p = nodes_sum[2*K+1, node]
			λw_p = lambda * w_p

			gain_p = zero(T)
			if L <: EvoTrees.GradientRegression
				if K == 1
					g_p = nodes_sum[1, node]
					h_p = nodes_sum[2, node]
					denom_p = h_p + λw_p + L2
					denom_p = denom_p < eps ? eps : denom_p
					gain_p = g_p^2 / denom_p / 2
				else
					for k in 1:K
						g_p = nodes_sum[k, node]
						h_p = nodes_sum[K+k, node]
						denom_p = h_p + λw_p + L2
						denom_p = denom_p < eps ? eps : denom_p
						gain_p += g_p^2 / denom_p / 2
					end
				end
			elseif L <: EvoTrees.MLE2P
				g1 = nodes_sum[1, node]
				g2 = nodes_sum[2, node]
				h1 = nodes_sum[3, node]
				h2 = nodes_sum[4, node]
				denom1 = h1 + λw_p + L2
				denom2 = h2 + λw_p + L2
				denom1 = denom1 < eps ? eps : denom1
				denom2 = denom2 < eps ? eps : denom2
				gain_p = (g1^2 / denom1 + g2^2 / denom2) / 2
			elseif L == EvoTrees.MLogLoss
				for k in 1:K
					gk = nodes_sum[k, node]
					hk = nodes_sum[K+k, node]
					denom = hk + λw_p + L2
					denom = denom < eps ? eps : denom
					gain_p += gk^2 / denom / 2
				end
			elseif (L == EvoTrees.MAE || L == EvoTrees.Quantile)
				gain_p = zero(T)
			elseif L <: EvoTrees.Cred
				μp = nodes_sum[1, node] / w_p
				VHM = μp^2
				EVPV = nodes_sum[2, node] / w_p - VHM
				EVPV = EVPV < eps ? eps : EVPV
				Zp = VHM / (VHM + EVPV)
				gain_p = Zp * abs(nodes_sum[1, node]) / (1 + L2 / w_p)
			end

			g_best = T(-Inf)
			b_best = Int32(0)
			f_best = Int32(0)

			for j_idx in 1:length(js)
				f = js[j_idx]
				is_numeric = feattypes[f]
				constraint = monotone_constraints[f]

				if K == 1
					acc1 = zero(T)
					acc2 = zero(T)
					accw = zero(T)
				else
					for kk in 1:(2*K+1)
						sums_temp[kk, n_idx] = zero(T)
					end
				end

				b_max = is_numeric ? (nbins - 1) : nbins
				for b in 1:b_max
					if K == 1
						if is_numeric
							acc1 += h∇[1, b, f, node]
							acc2 += h∇[2, b, f, node]
							accw += h∇[3, b, f, node]
						else
							acc1 = h∇[1, b, f, node]
							acc2 = h∇[2, b, f, node]
							accw = h∇[3, b, f, node]
						end
						w_l = accw
						w_r = w_p - w_l
						(w_l < min_weight || w_r < min_weight) && continue

						g_val = zero(T)
						if L <: EvoTrees.GradientRegression
							g_l = acc1
							h_l = acc2
							g_r = nodes_sum[1, node] - g_l
							h_r = nodes_sum[2, node] - h_l
							d_l = h_l + lambda * w_l + L2
							d_r = h_r + lambda * w_r + L2
							d_l = d_l < eps ? eps : d_l
							d_r = d_r < eps ? eps : d_r
							g_val = (g_l^2 / d_l + g_r^2 / d_r) / 2 - gain_p

							if constraint != 0
								pred_l = -g_l / d_l
								pred_r = -g_r / d_r
								if (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
									continue
								end
							end
						elseif L == EvoTrees.MAE
							μp = nodes_sum[1, node] / w_p
							μl = acc1 / w_l
							μr = (nodes_sum[1, node] - acc1) / w_r
							d_l = 1 + lambda + L2 / w_l
							d_r = 1 + lambda + L2 / w_r
							d_l = d_l < eps ? eps : d_l
							d_r = d_r < eps ? eps : d_r
							g_val = abs(μl - μp) * w_l / d_l + abs(μr - μp) * w_r / d_r
						elseif L == EvoTrees.Quantile
							μp = nodes_sum[1, node] / w_p
							μl = acc1 / w_l
							μr = (nodes_sum[1, node] - acc1) / w_r
							d_l = 1 + lambda + L2 / w_l
							d_r = 1 + lambda + L2 / w_r
							d_l = d_l < eps ? eps : d_l
							d_r = d_r < eps ? eps : d_r
							g_val = abs(μl - μp) * w_l / d_l + abs(μr - μp) * w_r / d_r
						elseif L <: EvoTrees.Cred
							μp = nodes_sum[1, node] / w_p
							VHM_p = μp^2
							EVPV_p = nodes_sum[2, node] / w_p - VHM_p
							EVPV_p = EVPV_p < eps ? eps : EVPV_p
							Zp = VHM_p / (VHM_p + EVPV_p)
							μl = acc1 / w_l
							VHM_l = μl^2
							EVPV_l = acc2 / w_l - VHM_l
							EVPV_l = EVPV_l < eps ? eps : EVPV_l
							Zl = VHM_l / (VHM_l + EVPV_l)
							g_l = Zl * abs(acc1) / (1 + L2 / w_l)
							μr = (nodes_sum[1, node] - acc1) / w_r
							VHM_r = μr^2
							EVPV_r = (nodes_sum[2, node] - acc2) / w_r - VHM_r
							EVPV_r = EVPV_r < eps ? eps : EVPV_r
							Zr = VHM_r / (VHM_r + EVPV_r)
							g_r = Zr * abs(nodes_sum[1, node] - acc1) / (1 + L2 / w_r)
							g_val = g_l + g_r - Zp * abs(nodes_sum[1, node]) / (1 + L2 / w_p)
						end
					else
						if is_numeric
							for kk in 1:(2*K+1)
								sums_temp[kk, n_idx] += h∇[kk, b, f, node]
							end
						else
							for kk in 1:(2*K+1)
								sums_temp[kk, n_idx] = h∇[kk, b, f, node]
							end
						end

						w_l = sums_temp[2*K+1, n_idx]
						w_r = w_p - w_l
						(w_l < min_weight || w_r < min_weight) && continue

						if L == EvoTrees.MLogLoss
						elseif constraint != 0
							g_l1 = sums_temp[1, n_idx]
							h_l1 = sums_temp[K+1, n_idx]
							g_r1 = nodes_sum[1, node] - g_l1
							h_r1 = nodes_sum[K+1, node] - h_l1
							d1_l = h_l1 + lambda * w_l + L2
							d1_r = h_r1 + lambda * w_r + L2
							d1_l = d1_l < eps ? eps : d1_l
							d1_r = d1_r < eps ? eps : d1_r
							pred_l = -g_l1 / d1_l
							pred_r = -g_r1 / d1_r
							if (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
								continue
							end
						end

						g_val = zero(T)
						for k in 1:K
							g_l = sums_temp[k, n_idx]
							h_l = sums_temp[K+k, n_idx]
							g_r = nodes_sum[k, node] - g_l
							h_r = nodes_sum[K+k, node] - h_l
							d_l = h_l + lambda * w_l + L2
							d_r = h_r + lambda * w_r + L2
							d_l = d_l < eps ? eps : d_l
							d_r = d_r < eps ? eps : d_r
							g_val += (g_l^2 / d_l + g_r^2 / d_r) / 2
						end
						g_val -= gain_p
					end

					if g_val > g_best
						g_best = g_val
						b_best = Int32(b)
						f_best = Int32(f)
					end
				end
			end

			gains[n_idx] = g_best
			bins[n_idx] = b_best
			feats[n_idx] = f_best
		end
	end
end

@kernel function clear_hist_kernel!(h∇, @Const(active_nodes), n_active)
	idx = @index(Global, Linear)
	n_elements = size(h∇, 1) * size(h∇, 2) * size(h∇, 3)
	total = n_elements * n_active

	if idx <= total
		node_idx = (idx - 1) ÷ n_elements + 1
		element_idx = (idx - 1) % n_elements

		@inbounds node = active_nodes[node_idx]
		if node > 0
			k = element_idx % size(h∇, 1) + 1
			b = (element_idx ÷ size(h∇, 1)) % size(h∇, 2) + 1
			j = element_idx ÷ (size(h∇, 1) * size(h∇, 2)) + 1
			h∇[k, b, j, node] = zero(eltype(h∇))
		end
	end
end

@kernel function clear_mask_kernel!(mask)
	idx = @index(Global)
	if idx <= length(mask)
		mask[idx] = 0
	end
end

@kernel function mark_active_nodes_kernel!(mask, @Const(active_nodes))
	idx = @index(Global)
	if idx <= length(active_nodes)
		node = active_nodes[idx]
		if node > 0 && node <= length(mask)
			mask[node] = 1
		end
	end
end

# Count raw number of observations per node for the current is/nidx mapping
@kernel function count_nodes_kernel!(node_counts, @Const(nidx), @Const(is))
	idx = @index(Global)
	if idx <= length(is)
		obs = is[idx]
		node = nidx[obs]
		if node > 0 && node <= length(node_counts)
			Atomix.@atomic node_counts[node] += 1
		end
	end
end

# Build histograms for a set of active nodes
function update_hist_gpu!(
	h∇, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
	feattypes, monotone_constraints, K, L2, sums_temp, target_mask, backend,
)
	n_active = length(active_nodes)

	if sums_temp === nothing && K > 1
		sums_temp = similar(nodes_sum_gpu, 2 * K + 1, max(n_active, 1))
	elseif K == 1
		sums_temp = similar(nodes_sum_gpu, 1, 1)
	end

	clear_mask_kernel!(backend)(target_mask; ndrange = length(target_mask), workgroupsize = 256)
	KernelAbstractions.synchronize(backend)

	mark_active_nodes_kernel!(backend)(target_mask, active_nodes; ndrange = n_active, workgroupsize = 256)
	KernelAbstractions.synchronize(backend)

	if n_active > 0
		clear_hist_kernel!(backend)(
			h∇, active_nodes, n_active;
			ndrange = n_active * size(h∇, 1) * size(h∇, 2) * size(h∇, 3),
			workgroupsize = 256,
		)
		KernelAbstractions.synchronize(backend)
	end

	chunk_size = 64
	n_obs_chunks = cld(length(is), chunk_size)
	num_threads = length(js) * n_obs_chunks

	hist_kernel_f! = hist_kernel!(backend)
	hist_kernel_f!(
		h∇, ∇, x_bin, nidx, js, is, K, chunk_size, target_mask;
		ndrange = num_threads,
		workgroupsize = 256,
	)
	KernelAbstractions.synchronize(backend)
end

