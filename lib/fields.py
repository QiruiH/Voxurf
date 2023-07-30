import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# This implementation is borrowed from NR-NeRF: https://github.com/facebookresearch/nonrigid_nerf
# N.B. it is significantly modified
class BendingNetwork(nn.Module):
    def __init__(self,
                 latents,
                 d_hidden,
                 n_layers,
                 latent_dim=64,
                 zero_init=True,
                 exact_viewdirs=False,
                 exact_divergence=False,
                 absolute_offset_loss=False,
                 one_pow_absolute=False,
                 template_frames=-1):
        super(BendingNetwork, self).__init__()

        # latents就是传入的需要训练的隐向量
        self.latents = latents
        self.n_frames = len(self.latents)
        # 64
        self.ray_bending_latent_size = self.latents[0].shape[-1]
        self.exact_viewdirs = exact_viewdirs
        self.exact_divergence = exact_divergence
        self.absolute_offset_loss = absolute_offset_loss
        self.one_pow_absolute = one_pow_absolute
        # -1
        self.template_frames = template_frames

        self.input_ch = 3
        self.output_ch = 3
        self.activation_function = F.relu  # F.relu, torch.sin
        # 64
        self.hidden_dimensions = d_hidden
        # 5
        self.network_depth = n_layers
        self.skips = []
        use_last_layer_bias = False

        # 网络为一个五层的MLP
        self.network = nn.ModuleList(
            [
                nn.Linear(
                    self.input_ch + self.ray_bending_latent_size,
                    self.hidden_dimensions,
                )
            ]
            + [
                nn.Linear(
                    self.input_ch + self.hidden_dimensions, self.hidden_dimensions
                )
                if i + 1 in self.skips
                else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
                for i in range(self.network_depth - 2)
            ]
            + [
                nn.Linear(
                    self.hidden_dimensions, self.output_ch, bias=use_last_layer_bias
                )
            ]
        )

        # initialize weights
        with torch.no_grad():
            for i, layer in enumerate(self.network[:-1]):
                if self.activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(layer.bias)

            '''
            self.network[-1] = Linear(in_features=64, out_features=3, bias=False)
            即有3 * 64个参数需要更新：
            ipdb> self.network[-1].weight.data.shape
            torch.Size([3, 64])
            '''
            # initialize final layer to zero weights to start out with straight rays
            self.network[-1].weight.data *= 0.0
            if use_last_layer_bias:
                self.network[-1].bias.data *= 0.0
    
    
    def forward(self,
                input_pts, # not positionally encoded!
                frame):
        
        # input_pts是从原图中采样得到的真实的pts, 此时input_pts为三维
        # 我们不需要这部分，我们自己的input_pts就是2
        
        # 这里有问题，dvgo采样和后面不一样，它是每条ray均衡采样
        # dvgo stage: input_pts [8192, 396, 3]
        is_dvgo = (len(input_pts.shape) == 3)

        details = {"input_pts": input_pts}

        if is_dvgo:
            n_rays, n_samples, _ = input_pts.shape
            input_pts = input_pts.view(n_rays * n_samples, 3)

        # fully-connected network regresses offset
        # 拿出相应帧的隐向量
        latent = self.latents[frame]
        # 扩展到 [n_rays*n_samples, 64]，即相当于每个点对应一个64的向量
        repeat_latent = latent[None, ...].expand(input_pts.shape[0], -1)

        # __import__('ipdb').set_trace()

        # [n_rays*n_samples, 3+64]
        h = torch.cat([input_pts, repeat_latent], -1)

        for i, layer in enumerate(self.network):
            # 遍历bendingnetwork的每一层
            # 将输入过一遍网络
            h = layer(h)

            # SIREN
            if self.activation_function.__name__ == "sin" and i == 0:
                h *= 30.0

            if (
                i != len(self.network) - 1
            ):  # no activation function after last layer (Relu prevents backprop if the input is zero & need offsets in positive and negative directions)
                # 最后一层网络不过激活函数
                h = self.activation_function(h)

            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # 将最终的输出赋值给offsets
        # [n_rays * n_samples, 3]
        offsets = h
        # 形变之后的点 → 原始点 + 偏移量
        bent_points = input_pts + offsets

        if is_dvgo:
            offsets = offsets.view(n_rays, n_samples, 3)
            bent_points = bent_points.view(n_rays, n_samples, 3)

        details["offsets"] = offsets
        details["bent_pts"] = bent_points
            
        return bent_points, details
        
    
    def extrapolate_scene_flow(self,
                               pts,
                               verts,
                               f1,
                               f2,
                               extrap_scale, # aka interp_scale
                               falloff_scale,
                               details=None):
        """
        Computes the extrapolated scene flow at pts from frame f1 to frame f2 given the 
        mesh vertices in verts. Gaussian kernels are used for weighted average when extrapolating
        (extrap_scale) and falloff as samples fall further from the mesh (falloff_width).
        pts.shape = n_samples x 3
        verts.shape = n_frames x n_vertices x 3
        """
        n_samples = pts.shape[0]

        def kernel(x, scale): # point-wise Gaussian-ish kernel
            return torch.exp(-scale * torch.pow(x, 2.))

        with torch.no_grad():
            scene_flow_pts = torch.Tensor(verts[f1]).to(pts.get_device())
            scene_flow_vecs = torch.Tensor(verts[f2] - verts[f1]).to(pts.get_device())
            if details is not None:
                details['proxy_pts'] = verts[f1]
                details['proxy_flow_vecs'] = verts[f2] - verts[f1]

            # Compute the weight and scene flow for all mesh vertices
            n_vertices = scene_flow_pts.shape[0]
            scene_flow_pts = scene_flow_pts.unsqueeze(0).repeat(n_samples, 1, 1) # n_samples x n_vertices x 3
            repeat_pts = pts.unsqueeze(1).repeat(1, n_vertices, 1) # n_samples x n_vertices x 3
            distances = repeat_pts - scene_flow_pts # n_samples x n_vertices x 3
            vert_weights = kernel(torch.norm(distances, dim=-1), extrap_scale) # n_samples x n_vertices
            scene_flow_weights = torch.sum(vert_weights, 1) # n_samples

            # Perform the weighted summation over all mesh vertices
            scene_flow_estimate = (
                vert_weights.unsqueeze(-1).repeat(1, 1, 3)
                * scene_flow_vecs.unsqueeze(0).repeat(n_samples, 1, 1)
            ) # n_samples x n_vertices x 3
            scene_flow_estimate = torch.sum(scene_flow_estimate, 1) # n_samples x 3

            # Complete the convex combination by normalizing by the summed weights
            # N.B. summed weights can be very small and cause division by zero
            scene_flow_weights = scene_flow_weights.unsqueeze(-1).repeat(1, 3)
            scene_flow_estimate = torch.div(scene_flow_estimate, scene_flow_weights)
            scene_flow_estimate[scene_flow_weights < 1e-8] = 0.

            # Estimated scene flow falls off as the sample is further from the mesh
            min_dists, _ = torch.min(torch.norm(distances, dim=-1), dim=1) # n_samples
            scene_flow_estimate = kernel(min_dists, falloff_scale).unsqueeze(-1).repeat(1, 3) * scene_flow_estimate

        return scene_flow_estimate # n_samples x 3

    def gaussian_sample_about_verts(self,
                                    verts,
                                    scale,
                                    n_points,
                                    scene_min_pt,
                                    scene_max_pt,
                                    single_gaussian=False):
        from scipy.stats import multivariate_normal
        cov = np.eye(3) / (2. * scale)
        if single_gaussian:
            pts = multivariate_normal.rvs(mean=np.mean(verts, axis=0), cov=cov, size=n_points)
        else:
            centers_i = np.random.randint(0, len(verts), size=n_points)
            centers = verts[centers_i]
            # TODO could be implemented more efficiently!
            pts = np.empty((n_points, 3))
            for i in range(len(pts)):
                pts[i] = multivariate_normal.rvs(mean=centers[i], cov=cov)
        np.clip(pts, scene_min_pt, scene_max_pt)
        return pts

    def compute_flow_loss(self,
                          frame,
                          verts,
                          batch_size,
                          scene_min_pt,
                          scene_max_pt,
                          extrap_scale, # aka interp_scale
                          falloff_scale,
                          frame_radius=-1,
                          template_frame=-1,
                          details=None):
        """
        Computes the flow loss.
        verts.shape = n_frames x n_vertices x 3
        """
        if details is None:
            # Determine the second frame for which to consider the scene flow
            f1 = frame.cpu().numpy()
            f2 = f1
            while f2 == f1:
                if frame_radius < 0:
                    if template_frame < 0:
                        # Random over all frames
                        f2 = np.random.randint(self.n_frames)
                    else:
                        # Random over deforming frames
                        f2 = np.random.randint(template_frame, self.n_frames)
                else:
                    # Random within radius and valid selection
                    f2 = np.random.randint(
                        max(0, f1 - frame_radius),
                        min(self.n_frames, f1 + frame_radius)
                    )
        else:
            f1 = details['f1']
            f2 = details['f2']

        # Sample the points for which to compute the flow loss
        pts = self.gaussian_sample_about_verts(
            verts[f1],
            extrap_scale,
            batch_size,
            scene_min_pt,
            scene_max_pt
        )
        if details is not None:
            details['pts'] = pts
        pts = torch.Tensor(pts).cuda() # n_samples x 3
        n_samples = pts[0]

        # Compute the interpolated scene flow for these points
        scene_flow_vecs = self.extrapolate_scene_flow(
            pts,
            verts,
            f1,
            f2,
            extrap_scale,
            falloff_scale,
            details
        ) # n_samples x 3
        if details is not None:
            details['scene_flow_vecs'] = scene_flow_vecs.detach().cpu().numpy()

        # Invoke ray bender on points after scene flow for secondary frame
        _, f2_details = self.forward(
            pts + scene_flow_vecs,
            f2
        )
        f2_deform_vecs = f2_details['offsets'] # n_samples x 3
        if details is not None:
            details['f2_deform_vecs'] = f2_deform_vecs.detach().cpu().numpy()

        # Invoke ray bender on points for primary frame
        _, f1_details = self.forward(pts, f1)
        f1_deform_vecs = f1_details['offsets'] # n_samples x 3
        if details is not None:
            details['f1_deform_vecs'] = f1_deform_vecs.detach().cpu().numpy()

        # Scene flow plus frame 2 deformation should equal frame 1 deformation
        # ie. both points deform to the same canonical point
        losses = torch.norm(
            scene_flow_vecs + f2_deform_vecs - f1_deform_vecs,
            dim=-1
        )
        return torch.mean(losses)

    def compute_viewdirs(self,
                         dirs,
                         details,
                         frame):
        if self.exact_viewdirs:
            return self._exact_nonrigid_viewdirs(
                details["input_pts"],
                dirs,
                frame
            )
        else:
            # 自己的数据进了这个分支
            return self._viewdirs_via_finite_differences(
                details["bent_pts"]
            )

    def compute_offset_loss(self,
                            details,
                            weights,
                            frame):
        input_pts = details["input_pts"] # shape: n_rays, n_samples, 3
        offsets = details["offsets"] # shape: n_rays, n_samples, 3
        
        # __import__('ipdb').set_trace()

        # 若长度为2，既然是每条ray计算一个loss，我们不如变成[n, 1, 3]的格式
        # 其实是1条ray还是n条rays应该都不是很影响
        if len(input_pts.shape) == 2:
            input_pts = input_pts.unsqueeze(0)
            input_pts = input_pts.reshape(input_pts.shape[1], 1, 3)
            offsets = offsets.unsqueeze(0)
            offsets = offsets.reshape(offsets.shape[1], 1, 3)
        
        if len(input_pts.shape) - len(weights.shape) != 1:
            weights = weights.unsqueeze(0)
            weights = weights.reshape(weights.shape[1], 1)
        '''
        coarse init阶段，也就是正常阶段
        weights [1024, 352]
        input_pts [1024, 352, 3]
        '''

        '''
        还是有问题，weights比input_pts还要少，因为我们coarse或者fine的过程已经筛过了
        '''
        # Remove background samples since those aren't bent
        weights = weights.detach()[:, :offsets.shape[1]] # shape: n_rays, n_samples
        n_rays = input_pts.shape[0]
        # Use absolute offset penalizer for template frames
        # N.B. template_frames=-1 for sequences w/o template part
        if self.absolute_offset_loss or (frame < self.template_frames):
            offsets = offsets.view(-1, 3)
            weights = weights.view(-1)
            offset_loss = torch.mean(
                (weights * torch.pow(
                    torch.norm(offsets, dim=-1),
                    (1. if self.one_pow_absolute else 2.) # N.B. no rigidity loss in this implementation!
                )).view(n_rays, -1),
                dim=-1
            ) # shape: n_rays
        else:
            # 这是要一条ray计算一个offset
            offset_loss = torch.zeros(n_rays)
            for i in [-1, 1]: # direct neighbouring frames
                neighbour = frame + i
                if neighbour < 0 or neighbour >= self.n_frames:
                    continue
                _, neighbour_details = self.forward(input_pts, neighbour)
                neighbour_offsets = neighbour_details["offsets"].detach()
                offset_loss += torch.mean(
                    (weights * torch.pow(
                        torch.norm(
                            offsets - neighbour_offsets,
                            dim=-1
                        ),
                        2.
                    )),
                    dim=-1
                ) # shape: n_rays
        return offset_loss

    def compute_divergence_loss(self,
                                details,
                                weights,
                                frame):
        input_pts = details["input_pts"] # shape: n_rays, n_samples, 3
        # Remove background samples since those aren't bent
        # 若长度为2，既然是每条ray计算一个loss，我们不如变成[n, 1, 3]的格式
        # 其实是1条ray还是n条rays应该都不是很影响
        if len(input_pts.shape) == 2:
            input_pts = input_pts.unsqueeze(0)
            input_pts = input_pts.reshape(input_pts.shape[1], 1, 3)
        if len(input_pts.shape) - len(weights.shape) != 1:
            weights = weights.unsqueeze(0)
            weights = weights.reshape(weights.shape[1], 1)
        
        weights = weights[:, :input_pts.shape[1]].contiguous() # shape: n_rays, n_samples
        n_rays = input_pts.shape[0]
        return self._compute_divergence_loss(
            input_pts.view(-1, 3),
            frame,
            n_rays,
            weights=weights
        ) # shape: n_rays

    def map_into_frame(self, can_pts, frame):
        # Construct initial guess for points in this frame
        with torch.no_grad():
            _, can_bend_details = self.forward(can_pts, frame)
            can_bend = can_bend_details['offsets'].squeeze(0) # remove virtual ray axis
            initial_guess = can_pts - can_bend
        # Optimizer setup
        frame_pts = initial_guess.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([frame_pts], lr=0.01)
        loss_fn = torch.nn.MSELoss()
        # Solve the optimization problem
        num_iters = 2500
        for i in range(num_iters):
            _, bend_details = self.forward(frame_pts, frame)
            bend = bend_details['offsets'].squeeze(0)
            mapped_into_can = frame_pts + bend
            loss = loss_fn(mapped_into_can, can_pts)
            loss.backward()
            opt.step()
            opt.zero_grad()
        return frame_pts

    def _viewdirs_via_finite_differences(self,
                                        input_pts): # n_rays, n_samples, 3
        # 为了保证正确性，若input_pts是2维的，我们unsqueeze一下
        need_unsqueeze = (len(input_pts.shape) == 2)
        if need_unsqueeze:
            input_pts = input_pts.unsqueeze(0)

        eps = 0.000001
        difference_type = "backward"
        if difference_type == "central":
            # central differences (except for first and last sample since one neighbor is missing for them)
            unnormalized_central_differences = (
                input_pts[:, 2:, :] - input_pts[:, :-2, :]
            )  # rays x (samples-2) x 3
            central_differences = unnormalized_central_differences / (
                torch.norm(unnormalized_central_differences, dim=-1, keepdim=True) + eps
            )
            # fill in first and last sample by duplicating neighboring direction
            input_views = torch.cat(
                [
                    central_differences[:, 0, :].view(-1, 1, 3),
                    central_differences,
                    central_differences[:, -1, :].view(-1, 1, 3),
                ],
                axis=1,
            )  # rays x samples x 3
        elif difference_type == "backward":
            # 自己的数据走了这个分支
            unnormalized_backward_differences = (
                input_pts[:, 1:, :] - input_pts[:, :-1, :]
            )  # rays x (samples-1) x 3. 0-th sample has no direction.
            backward_differences = unnormalized_backward_differences / (
                torch.norm(unnormalized_backward_differences, dim=-1, keepdim=True)
                + eps
            )
            # fill in first sample by duplicating neighboring direction
            input_views = torch.cat(
                [backward_differences[:, 0, :].view(-1, 1, 3), backward_differences],
                axis=1,
            )  # rays x samples x 3
        else:
            raise RuntimeError("invalid difference_type")

        if need_unsqueeze:
            input_views = input_views.squeeze(0)
            
        return input_views

    # from FFJORD github code: https://github.com/rtqichen/ffjord
    def _get_minibatch_jacobian(self, y, x):
        """Computes the Jacobian of y wrt x assuming minibatch-mode.
        Args:
        y: (N, ...) with a total of D_y elements in ...
        x: (N, ...) with a total of D_x elements in ...
        Returns:
        The minibatch Jacobian matrix of shape (N, D_y, D_x)
        """
        assert y.shape[0] == x.shape[0]
        y = y.view(y.shape[0], -1)

        # Compute Jacobian row by row.
        jac = []
        for j in range(y.shape[1]):
            dy_j_dx = torch.autograd.grad(
                y[:, j],
                x,
                torch.ones_like(y[:, j], device=y.get_device()),
                retain_graph=True,
                create_graph=True,
            )[0].view(x.shape[0], -1)
            jac.append(torch.unsqueeze(dy_j_dx, 1))
        jac = torch.cat(jac, 1)
        return jac

    def _exact_nonrigid_viewdirs(self,
                                 initial_input_pts,
                                 unbent_ray_direction,
                                 frame):
        straight_pts = initial_input_pts.reshape(-1, 3)
        straight_pts.requires_grad = True

        # compute Jacobian
        with torch.enable_grad():  # necessay to work properly in no_grad() mode
            # TODO should be possible without re-doing this bend (NR-NeRF does that)
            bent_pts, _ = self.forward(straight_pts, frame)
            jacobian = self._get_minibatch_jacobian(
                bent_pts, straight_pts
            )  # shape: N x 3 x 3. N x ouptut_dims x input_dims

        # compute directional derivative: J * d
        direction = unbent_ray_direction.reshape(-1, 3, 1)  # N x 3 x 1
        directional_derivative = torch.matmul(jacobian, direction)  # N x 3 x 1

        # normalize to unit length
        directional_derivative = directional_derivative.view(-1, 3)
        normalized_directional_derivative = (
            directional_derivative
            / torch.norm(directional_derivative, dim=-1, keepdim=True)
            + 0.000001
        )

        input_views = normalized_directional_derivative.view(
            -1, 3
        )  # rays * samples x 3

        return input_views

    # from FFJORD github code: https://github.com/rtqichen/ffjord
    def _divergence_exact(self, input_points, offsets_of_inputs):
        # requires three backward passes instead one like divergence_approx
        jac = self._get_minibatch_jacobian(offsets_of_inputs, input_points)
        diagonal = jac.view(jac.shape[0], -1)[:, :: (jac.shape[1]+1)]
        return torch.sum(diagonal, 1)

    # from FFJORD github code: https://github.com/rtqichen/ffjord
    def _divergence_approx(self, input_points, offsets_of_inputs):  # , as_loss=True):
        # avoids explicitly computing the Jacobian
        e = torch.randn_like(offsets_of_inputs, device=offsets_of_inputs.get_device())
        e_dydx = torch.autograd.grad(
            offsets_of_inputs,
            input_points,
            e,
            retain_graph=True,
            create_graph=True
        )[0]
        e_dydx_e = e_dydx * e
        approx_tr_dydx = e_dydx_e.view(offsets_of_inputs.shape[0], -1).sum(dim=1)
        return approx_tr_dydx

    def _compute_divergence_loss(self,
                                 input_points,
                                 frame,
                                 n_rays,
                                 chunk=16*1024,
                                 weights=None,
                                 backprop_into_weights=True):
        divergence_fn = self._divergence_exact if self.exact_divergence else self._divergence_approx

        input_points.requires_grad = True

        def divergence_wrapper(subtensor):
            _, details = self.forward(subtensor, frame)
            offsets = details["offsets"]
            return divergence_fn(subtensor, offsets)

        divergence_loss = torch.cat(
            [
                divergence_wrapper(input_points[i : i + chunk, None, :])
                for i in range(0, input_points.shape[0], chunk)
            ],
            dim=0,
        )

        divergence_loss = torch.abs(divergence_loss)
        divergence_loss = divergence_loss ** 2
        
        if weights is not None:
            if not backprop_into_weights:
                weights = weights.detach()
            divergence_loss = weights.view(-1) * divergence_loss
        # don't take mean, instead reshape to N_rays x samples, take mean across samples, return shape N_rays
        return torch.mean(divergence_loss.view(n_rays, -1), dim=-1)
