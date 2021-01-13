# -*- coding: utf-8 -*-

# The code in this file is adapted from https://github.com/ganguli-lab/grid-pattern-formation
# The adaptations made include a conversion to torch

# {Sorscher  B.  Mel  G.  Ganguli  S. \& Ocko  S. A unified theory for the origin of grid cells through the lens of pattern formation. \textit{Advances in Neural Information Processing Systems} (2019).}

import numpy as np
import torch as to
import matplotlib.pyplot as plt
import torch.nn.functional as F

import scipy

test_place_decoders = False


class PlaceCells(object):
    def __init__(self, options):
        self.Nx = options.Nx  # number of spatial indices
        self.Np = options.Np  # number of place cells
        self.res = options.res
        self.sigma = options.place_cell_rf # width of place cell center tuning curve (m)
        self.surround_scale = options.surround_scale # if DoG, ratio of sigma2^2 to sigma1^2
        self.box_width = options.box_width  # width of training environment
        self.box_height = options.box_height  # height of training environment
        self.periodic = options.periodic # trajectories with periodic boundary conditions
        self.DoG = options.DoG  # use difference of gaussians tuning curves
        self.gauss_norm = options.gauss_norm # use analytic normalization in gaussian function (vs softmax across pop)
        self.norm_cov = options.norm_cov

        # discretize space
        self.coordsx = np.linspace(-self.box_width/2, self.box_width/2, self.res)
        self.coordsy = np.linspace(-self.box_height/2, self.box_height/2, self.res)
        self.grid_x, self.grid_y = np.meshgrid(self.coordsx, self.coordsy)
        self.grid = np.stack([self.grid_x.ravel(), self.grid_y.ravel()]).T

        # Randomly tile place cell centers across environment
        to.manual_seed(0)
        usx = to.FloatTensor(self.Np,).uniform_(-self.box_width / 2, self.box_width / 2)
        usy = to.FloatTensor(self.Np,).uniform_(
            -self.box_height / 2, self.box_height / 2
        )
        self.us = to.stack([usx, usy], dim=-1)
        self.COV = 'EUCLID'

    def get_batch_activation(self, pos):
        """
        Get place cell activations for a given position.

        Args:
            pos: 2d position of shape [batch_size, sequence_length, 2].

        Returns:
            outputs: Place cell activations with shape [batch_size, sequence_length, Np].
        """
        pos = to.as_tensor(pos)
        d = to.abs(pos[:, :, np.newaxis, :] - self.us[np.newaxis, np.newaxis, ...])

        if self.periodic:
            dx = to.gather(input=d, dim=-1, index=0)
            dy = to.gather(input=d, dim=-1, index=1)
            dx = to.min(dx, self.box_width - dx)
            dy = to.min(dy, self.box_height - dy)
            d = to.stack([dx, dy], dim=-1)

        norm2 = to.sum(d ** 2, dim=-1)

        # Normalize place cell outputs with prefactor alpha=1/2/np.pi/self.sigma**2,
        # or, simply normalize with softmax, which yields same normalization on
        # average and seems to speed up training.
        if self.gauss_norm:
            np.seterr(under='ignore')
            outputs = np.exp(-norm2 / (2 * self.sigma ** 2))/(np.sqrt(2*np.pi)*self.sigma)
            outputs /= to.sum(input=outputs, dim=-1, keepdims=True)
            # outputs -= to.mean(input=outputs, dim=0, keepdims=True) # center
        else:
            outputs = F.softmax(-norm2 / (2 * self.sigma ** 2), dim=-1)

        if self.DoG:
            # Again, normalize with prefactor
            # beta=1/2/np.pi/self.sigma**2/self.surround_scale, or use softmax.
            if self.gauss_norm:
                outputs -= np.exp(-norm2 / (2 * self.surround_scale * self.sigma ** 2))/(np.sqrt(2*np.pi*self.surround_scale)*self.sigma)
            else:
                outputs -= F.softmax(
                    -norm2 / (2 * self.surround_scale * self.sigma ** 2), dim=-1
                )

            # Shift and scale outputs so that they lie in [0,1].
            outputs += to.abs(to.min(input=outputs, dim=-1, keepdims=True).values)
            outputs /= to.sum(input=outputs, dim=-1, keepdims=True)

        return outputs

    def get_activation(self, pos):
        """
        Get place cell activations for a given position.

        Args:
            pos: 2d position of shape.

        Returns:
            outputs: Place cell activations with shape [Np].
        """
        d = to.abs(to.FloatTensor(pos) - self.us)

        if self.periodic:
            # dx = to.gather(input=d, dim=-1, index=to.as_tensor(0))
            # dy = to.gather(input=d, dim=-1, index=to.as_tensor(1))
            dx = d[:,0]
            dy = d[:,1]
            dx = to.min(dx, self.box_width - dx)
            dy = to.min(dy, self.box_height - dy)
            d = to.stack([dx, dy], dim=-1)

        norm2 = to.sum(d ** 2, dim=-1)

        # Normalize place cell outputs with prefactor alpha=1/2/np.pi/self.sigma**2,
        # or, simply normalize with softmax, which yields same normalization on
        # average and seems to speed up training.
        if self.gauss_norm:
            np.seterr(under='ignore')
            outputs = np.exp(-norm2 / (2 * self.sigma ** 2))/(np.sqrt(2*np.pi)*self.sigma)
            # outputs /= to.sum(input=outputs, dim=-1, keepdims=True)
            # outputs -= to.mean(input=outputs, dim=0, keepdims=True) # center
        else:
            outputs = F.softmax(-norm2 / (2 * self.sigma ** 2), dim=-1)


        if self.DoG:
            # Again, normalize with prefactor
            # beta=1/2/np.pi/self.sigma**2/self.surround_scale, or use softmax.
            if self.gauss_norm:
                outputs -= np.exp(-norm2 / (2 * self.surround_scale * self.sigma ** 2))/(np.sqrt(2*np.pi*self.surround_scale)*self.sigma)
            else:
                outputs -= F.softmax(
                    -norm2 / (2 * self.surround_scale * self.sigma ** 2), dim=-1
                )

            # Shift and scale outputs so that they lie in [0,1] and sum to 1
            outputs -= outputs.min(dim=-1, keepdim=True)[0]
            outputs /= outputs.sum(dim=-1, keepdim=True)[0]

        return outputs

    def get_activation_matrix(self):
        """ returns (Nx,Np) matrix of place cell population activations """
        if not hasattr(self, 'P'):
            pos = np.array(
                np.meshgrid(
                    np.linspace(-self.box_width / 2, self.box_width / 2, self.res),
                    np.linspace(-self.box_height / 2, self.box_height / 2, self.res),
                )
            ).T.astype(np.float32)
            self.P = self.get_batch_activation(pos).reshape((-1, self.Np))
        return self.P

    def get_batch_nearest_cell_pos(self, activation, k=3):
        """
        Decode position using centers of k maximally active place cells.

        Args:
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].
        """
        _, idxs = to.topk(activation, k=k)
        pred_pos = to.mean(to.gather(input=self.us, index=idxs), dim=-2)
        return pred_pos

    def get_nearest_cell_pos(self, activation, k=3):
        """
        Decode position using centers of k maximally active place cells.

        Args:
            activation: Place cell activations of shape [Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [2].
        """
        _, idxs = to.topk(activation, k=k)
        pred_pos = to.mean(to.gather(input=self.us, index=idxs), dim=-2)
        return pred_pos

    def estimate_spatial_index(self, activation, k=3):
        '''
        Decode spatial index using centers of k maximally active place cells.

        Args:
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].
        '''
        pred_pos = self.get_nearest_cell_pos(activation=activation, k=k)
        pred_idx = self.pos2idx(pred_pos)

    def pos2idx(self, pos):
        """ returns the spatial index corresponding to the position pos """
        print('todo')

    def bayesian_spatial_decoder(self):
        """ inspired by ganguli/simoncelli """
        print('todo')


    def grid_pc(self, pc_outputs, res=32):
        """ Interpolate place cell outputs onto a grid"""
        coordsx = np.linspace(-self.box_width / 2, self.box_width / 2, res)
        coordsy = np.linspace(-self.box_height / 2, self.box_height / 2, res)
        grid_x, grid_y = np.meshgrid(coordsx, coordsy)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # Convert to numpy
        us_np = self.us.numpy()
        pc_outputs = pc_outputs.numpy().reshape(-1, self.Np)

        T = pc_outputs.shape[0]  # T vs transpose? What is T? (dim's?)
        pc = np.zeros([T, res, res])
        for i in range(len(pc_outputs)):
            gridval = scipy.interpolate.griddata(us_np, pc_outputs[i], grid)
            pc[i] = gridval.reshape([res, res])

        return pc

    def compute_covariance(self):
        """Compute spatial covariance matrix of place cell outputs"""
        pos = np.array(
            np.meshgrid(
                np.linspace(-self.box_width / 2, self.box_width / 2, self.res),
                np.linspace(-self.box_height / 2, self.box_height / 2, self.res),
            )
        ).T

        pos = pos.astype(np.float64)

        # Maybe specify dimensions here again?
        pc_outputs = self.get_batch_activation(pos)
        pc_outputs = to.reshape(pc_outputs, (-1, self.Np))

        C = pc_outputs @ pc_outputs.T
        self.SigmaP = C  # save covariance matrix for discrete/bounded domain

        Csquare = to.reshape(C, (self.res, self.res, self.res, self.res))
        # converts (Nx,Nx) graph covariance to (nx,nx,nx,nx) matrix where the first two indices index one position and the last two indices index the other position. That is,
        # Csquare[i,j,k,l] is the place covariance between positions (i,j) and (k,l)
        # Csquare[i,j] == Csquare[i,j,:,:] is the place covariance with respect position (i,j)
        # C = pc_outputs @ pc_outputs.T
        # Csquare = to.reshape(C, (res, res, res, res))

        Cmean = np.zeros([self.res, self.res])
        for i in range(self.res):
            for j in range(self.res):
                Cmean += np.roll(np.roll(Csquare[i, j], -i, axis=0), -j, axis=1)
        # the (-i,-j)-rolls periodically shift all covariances such that they are aligned at (0,0)
        # then the spatially aligned covariances are summed up
        # (resulting in a total covariance matrix, maybe should average)

        Cmean = np.roll(np.roll(Cmean, self.res // 2, axis=0), self.res // 2, axis=1)
        # this step (a (nx/2,nx/2) roll) centers the total spatial covariance
        if self.norm_cov:
            Cmean /= (self.res*self.res) # averaged
        self.Cmean = Cmean
        return Cmean


    def fft_covariance(self):
        if not hasattr(self, 'Cmean'):
            self.compute_covariance()
        self.Ctilde = np.fft.fft2(self.Cmean).real
        self.Ctilde[0,0] = 0
        return self.Ctilde


    def plot_receptive_fields(self, n_plot=6):
        P = self.get_activation_matrix()
        fig = plt.figure(figsize=(12, 4))
        for i in range(n_plot):
            plt.subplot(1, n_plot, i + 1)
            im0 = plt.imshow(P[:,i].reshape((self.res,self.res)), cmap="jet", vmin=0)
            plt.axis("off")
        fig.colorbar(im0)
        plt.suptitle("Place cell outputs", fontsize=16)
        plt.show()
        return plt.gcf()


    def plot_covariance(self, interpolation=None):
        if not hasattr(self,'Cmean'):
            self.compute_covariance()
        if not hasattr(self,'Ctilde'):
            self.fft_covariance()
        res = self.res

        plt.figure(figsize=(18,15))
        plt.subplot(331)
        SigmaSample = self.SigmaP.reshape((res,res,res,res))[res//2,res//2,:,:]
        plt.imshow(SigmaSample, cmap='jet', origin='lower', interpolation=interpolation)
        plt.colorbar()
        plt.axis('equal')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'$\left[P P^\top\right]_{x_{center}\cdot}$ [complete]', fontsize=20)

        plt.subplot(332)
        plt.imshow(self.Cmean, cmap='jet', origin='lower', interpolation=interpolation)
        plt.colorbar()
        plt.axis('equal')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'$\Sigma$ [averaged spatial covariance]', fontsize=20)

        plt.subplot(333)
        CtildeRoll = np.roll(np.abs(self.Ctilde), shift=[-res//2,-res//2], axis=[0,1])
        plt.imshow(CtildeRoll, cmap='Oranges', extent=(-res//2,res//2,-res//2,res//2), interpolation=interpolation)
        plt.colorbar()
        plt.axis('equal')
        plt.xlabel(r'$k_x$')
        plt.ylabel(r'$k_y$')
        plt.title(r'$\tilde \Sigma$ [rolled]', fontsize=20)

        # zooms
        plt.subplot(334)
        width = res//10
        idxs = np.arange(res//2-width+1, res//2+width)
        plt.imshow(SigmaSample[np.ix_(idxs,idxs)], origin='lower', cmap='jet', extent=(res//2-width, res//2+width,res//2-width, res//2+width), interpolation=interpolation)
        plt.colorbar()
        plt.axis('equal')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        # plt.axis('off')
        plt.title(r'$\left[P P^\top\right]_{x_{center}\cdot}$ [zoomed]', fontsize=20)
        plt.tight_layout()

        plt.subplot(335)
        plt.imshow(self.Cmean[np.ix_(idxs,idxs)], origin='lower', cmap='jet', extent=(res//2-width, res//2+width,res//2-width, res//2+width), interpolation=interpolation)
        plt.colorbar()
        plt.axis('equal')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        # plt.axis('off')
        plt.title(r'$\Sigma$ [zoomed]', fontsize=20)

        plt.subplot(336)
        idxs = np.arange(-width+1, width)
        plt.imshow(np.abs(self.Ctilde)[np.ix_(idxs,idxs)], cmap='Oranges', extent=(-width, width,-width, width), interpolation=interpolation)
        plt.colorbar()
        plt.axis('equal')
        plt.xlabel(r'$k_x$')
        plt.ylabel(r'$k_y$')
        # plt.axis('off')
        plt.title(r'$\tilde \Sigma$ [rolled/zoomed]', fontsize=20)

        # axial slices
        plt.subplot(337)
        x = res//2
        SigmaSampleSlice = self.SigmaP.reshape((res,res,res,res))[res//2,res//2,x,:]
        plt.plot(SigmaSampleSlice, c='k', lw=2)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\left[P P^\top\right]_{x_{center}\cdot}$ [$y=%i$]'%x)
        plt.title(r'$\left[P P^\top\right]_{x_{center}\cdot}$ [slice]', fontsize=20)
        plt.xlim([0,res-1])

        plt.subplot(338)
        plt.plot(range(res),self.Cmean[x,:], c='k', lw=2)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\Sigma$ [$y=%i$]'%x)
        plt.title(r'$\Sigma$ [slice]', fontsize=20)
        plt.xlim([0,res-1])

        plt.subplot(339)
        x = 0
        idxs = np.arange(-res//2,res//2)
        plt.plot(idxs, CtildeRoll[x+res//2,:], c='k', lw=2)
        plt.xlabel(r'$k_x$')
        plt.ylabel(r'$\tilde \Sigma$ [$y=%i$]'%x)
        plt.title(r'$\tilde \Sigma$ [slice]', fontsize=20)
        plt.xlim([idxs[0],idxs[-1]])

        plt.tight_layout()
        return plt.gcf()






class PredictivePlaceCells(PlaceCells):
    def __init__(self, options):
        assert hasattr(options,'Q'), 'generator required.'
        super(PredictivePlaceCells, self).__init__(options)
        self.COV = 'PREDICT'
        self.Q = to.as_tensor(options.Q)

    def get_activation_matrix(self):
        """ returns (Nx,Np) matrix of PREDICTIVE place cell population activations stored as member in self.PP """
        if not hasattr(self, 'PP'):
            pos = np.array(
                np.meshgrid(
                    np.linspace(-self.box_width / 2, self.box_width / 2, self.res),
                    np.linspace(-self.box_height / 2, self.box_height / 2, self.res),
                )
            ).T.astype(np.float64)
            self.P = self.get_batch_activation(pos).reshape((-1, self.Np))
            self.PP = self.Q@self.P  # (Nx,Np)-matrix of place cell activations
        return self.PP

    def get_predictive_matrix(self):
        """ returns (Nx,Nx) predictive matrix """
        return self.Q

    def compute_covariance(self, predmap=True):
        """
        predmap = False, compute spatial covariance matrix of PREDICTIVE place cell outputs
        predmap = True, compute spatial covariance matrix of PREDICTIVE map
        """
        if predmap:
            P = self.get_predictive_matrix()
        else:
            P = self.get_activation_matrix()

        C = P @ P.T
        self.SigmaP = C
        Csquare = to.reshape(C, (self.res, self.res, self.res, self.res))

        Cmean = np.zeros([self.res, self.res])
        for i in range(self.res):
            for j in range(self.res):
                Cmean += np.roll(np.roll(Csquare[i, j], -i, axis=0), -j, axis=1)

        Cmean = np.roll(np.roll(Cmean, self.res // 2, axis=0), self.res // 2, axis=1)  # center zero frequencies
        if  self.norm_cov:
            Cmean /= (self.res*self.res) # averaged
        self.Cmean = Cmean
        return Cmean

    def fft_covariance(self, predmap=True):
        """
        predmap = False, FFTs spatial covariance matrix of PREDICTIVE place cell outputs
        predmap = True, FFTs spatial covariance matrix of PREDICTIVE map
        """
        if not hasattr(self, 'Cmean'):
            self.compute_covariance(predmap=predmap)
        self.Ctilde = np.fft.fft2(self.Cmean).real
        self.Ctilde[0,0] = 0
        return self.Ctilde

    def plot_receptive_fields(self, n_plot=6):
        PP = self.get_activation_matrix()
        P = self.P
        fig = plt.figure(figsize=(2*n_plot, 9))
        for i in range(n_plot):
            plt.subplot(3, n_plot, i + 1)
            im0 = plt.imshow(P[:,i].reshape((self.res,self.res)), cmap="jet")
            plt.axis("off")
        fig.colorbar(im0)
        for i in range(n_plot):
            plt.subplot(3, n_plot, n_plot + i + 1)
            im1 = plt.imshow(PP[:,i].reshape((self.res,self.res)), cmap="jet")
            plt.axis("off")
        fig.colorbar(im1)
        for i in range(n_plot):
            plt.subplot(3, n_plot, 2*n_plot + i + 1)
            im2 = plt.imshow(self.Q[((self.res**2)//n_plot)*i,:].reshape((self.res,self.res)), cmap="jet")
            plt.axis("off")
        fig.colorbar(im2)
        plt.suptitle("Place cell outputs | predictive responses | predictive map", fontsize=16)
        plt.show()



if test_place_decoders:
    # %% test place decoder
    # plot place cells
    place_cells.plot_receptive_fields()

    P = to.as_tensor(place_cells.P).double()
    # print(to.matrix_rank(P))

    # pseudo-inverse is a left-inverse (decode cell from its receptive field) if Nx>=Np (linearly independent columns)
    Ppinv = to.pinverse(P.double())
    # pseudo-inverse is a right-inverse (decode position from population code) if Nx>=Np (linearly independent rows)
    # need to have equal/more cells Np than positions Nx to use pseudo-inverse as a decoder
    # Pdecode = (P.T) @ ((P@(P.T)).inverse()) # this seems inaccurate
    # Pdecode = to.pinverse(P.T).T # this is equal to to.pinverse(P) since pinv commutes with transpose
    Pdecode = to.pinverse(P.double())


    def generate_gaussian(options, idx=[10,10]):
        pos_range_min=-options.box_width/2
        pos_range_max=options.box_width/2
        res = options.res
        grid_x, grid_y = np.meshgrid(
                            np.linspace(pos_range_min, pos_range_max, res),
                            np.linspace(pos_range_min, pos_range_max, res)
                        )
        us = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        sigma = 0.12
        norm2 = np.linalg.norm(us, ord=2, axis=1)
        C = np.exp(-(norm2)**2 / (2.0 * sigma**2))
        C = C.reshape([res,res])
        return res


    def linear_sm_decoder(rhoP, P, beta=1e5):
        return F.softmax(beta*rhoP@P.T).reshape((options.res,options.res))

    def pseudo_inv_decoder(rhoP, P, thresh=5., beta=1e3):
        Ppinv = to.pinverse(P.double())
        S = rhoP@Ppinv
        # S = F.softmax(beta*S)
        S = F.threshold(input=S, threshold=S.max()/thresh, value=0)
        S = F.softmax(beta*S)
        # S /= S.sum()
        return S.reshape((options.res,options.res))



    rho_init = to.zeros((options.res,options.res)).double()
    rho_init[10,10] = 1.
    rho_init += 0.1
    rho_init /= rho_init.sum()
    rhoP = rho_init.flatten()@P

    rho_decode_sm = linear_sm_decoder(rhoP, P)
    rho_decode_pinv = pseudo_inv_decoder(rhoP, P)

    fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(20,10))
    vmin = 0; vmax = None
    # vmin = 0; vmax = 1
    im0 = axes[0][0].imshow(rho_init, cmap='gray', vmin=vmin, vmax=vmax)
    im1 = axes[0][1].imshow(rho_decode_sm, cmap='gray', vmin=vmin, vmax=vmax)
    im2 = axes[0][2].imshow(rho_decode_pinv, cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(im0, ax=axes[0][0], shrink=1)
    fig.colorbar(im1, ax=axes[0][1], shrink=1)
    fig.colorbar(im2, ax=axes[0][2], shrink=1)
    axes[0][0].set_title('true place density')
    axes[0][1].set_title('linear-softmax decoded place density')
    axes[0][2].set_title('pseudo-inverse decoded place density')

    im3 = axes[1][0].imshow(Ppinv@P, cmap='RdBu', norm=MidpointNormalize(midpoint=0, vmin=None, vmax=None))
    fig.colorbar(im3, ax=axes[1][0], shrink=1)
    axes[1][0].set_title('pseudoinverse(P) @ P [decodes place cell from firing map]')
    axes[1][0].set_xlabel(r'$n_P$')
    axes[1][0].set_ylabel(r'$n_P$')

    im4 = axes[1][1].imshow(P@P.T, cmap='RdBu', norm=MidpointNormalize(midpoint=0, vmin=None, vmax=None))
    fig.colorbar(im4, ax=axes[1][1], shrink=1)
    axes[1][1].set_title('P @ P.T [alignment between population responses ~ confusion matrix]')
    axes[1][1].set_xlabel(r'$n_X$')
    axes[1][1].set_ylabel(r'$n_X$')

    im5 = axes[1][2].imshow(P@Pdecode, cmap='RdBu', norm=MidpointNormalize(midpoint=0, vmin=None, vmax=None))
    fig.colorbar(im5, ax=axes[1][2], shrink=1)
    axes[1][2].set_title('P @ pseudoinverse(P) [decode spatial index from population code]')
    axes[1][2].set_xlabel(r'$n_X$')
    axes[1][2].set_ylabel(r'$n_X$')


#
# # %% NUMERICAL CHECKS OF CONVERSION FROM TENSORFLOW TO TORCH
# if __name__ == "__main__":
#     import tensorflow as tf
#     from place_cellsTF import PlaceCellsTF
#
#     res = 30
#
#     pos = np.array(
#         np.meshgrid(
#             np.linspace(-PP.box_width / 2, PP.box_width / 2, res),
#             np.linspace(-PP.box_height / 2, PP.box_height / 2, res),
#         )
#     ).T
#
#     pos = pos.astype(np.float32)
#
#     PP = PlaceCells(options=options)
#     PPTF = PlaceCellsTF(options=options)
#     PPTF.us = tf.convert_to_tensor(
#         PP.us.clone().detach().numpy()
#     )  # equalize place cell positions
#     pc_outputs = PP.get_batch_activation(pos)
#     pc_outputsTF = PPTF.get_activation(pos)
#
#     # Plot a few place cell outputs
#     pc_outputs = pc_outputs.reshape((-1, options.Np))
#     pc = PP.grid_pc(pc_outputs[::100], res=100)
#
#     plt.figure(figsize=(16, 2))
#     for i in range(8):
#         plt.subplot(1, 8, i + 1)
#         plt.imshow(pc[i], cmap="jet")
#         plt.axis("off")
#
#     plt.suptitle("Place cell outputs", fontsize=16)
#     plt.show()
#
#     pc_outputsTF = tf.reshape(pc_outputsTF, (-1, options.Np))
#     pc = PPTF.grid_pc(pc_outputsTF[::100], res=100)
#
#     plt.figure(figsize=(16, 2))
#     for i in range(8):
#         plt.subplot(1, 8, i + 1)
#         plt.imshow(pc[i], cmap="jet")
#         plt.axis("off")
#
#     plt.suptitle("Place cell outputs", fontsize=16)
#     plt.show()
#
#     # Visualize place cell covariance matrix
#     # examine covariance calculation
#     plt.figure()
#     SigmaP1 = PP.compute_covariance(res=32)
#     plt.imshow(SigmaP1, cmap=cmap_grid_code)
#     plt.colorbar()
#
#     plt.figure()
#     plt.imshow(model.SigmaP, cmap=cmap_grid_code)
#     plt.colorbar()
#
#     Cmean = PP.compute_covariance(res=30)
#     CmeanTF = PPTF.compute_covariance(res=30)
#
#     # Fourier transform
#     Ctilde = np.fft.fft2(Cmean)
#     CtildeTF = np.fft.fft2(CmeanTF)
#     Ctilde[0, 0] = 0
#     CtildeTF[0, 0] = 0
#
#     plt.figure(figsize=(8, 4))
#     plt.subplot(221)
#     plt.imshow(Cmean, cmap="jet", interpolation="gaussian")
#     plt.axis("off")
#     plt.title(r"$\Sigma$", fontsize=20)
#
#     plt.subplot(222)
#     plt.imshow(CmeanTF, cmap="jet", interpolation="gaussian")
#     plt.axis("off")
#     plt.title(r"$\Sigma$", fontsize=20)
#
#     plt.subplot(223)
#     width = 6
#     idxs = np.arange(-width + 1, width)
#     x2, y2 = np.meshgrid(np.arange(2 * width - 1), np.arange(2 * width - 1))
#     plt.scatter(
#         x2, y2, c=np.abs(Ctilde)[idxs][:, idxs], s=600, cmap="Oranges", marker="s"
#     )
#
#     plt.axis("equal")
#     plt.axis("off")
#     plt.title(r"$\tilde \Sigma$", fontsize=20)
#     plt.axis("off")
#
#     plt.subplot(224)
#     width = 6
#     idxs = np.arange(-width + 1, width)
#     x2, y2 = np.meshgrid(np.arange(2 * width - 1), np.arange(2 * width - 1))
#     plt.scatter(
#         x2, y2, c=np.abs(CtildeTF)[idxs][:, idxs], s=600, cmap="Oranges", marker="s"
#     )
#
#     plt.axis("equal")
#     plt.axis("off")
#     plt.title(r"$\tilde \Sigma$", fontsize=20)
#     plt.axis("off")
#     plt.tight_layout()
