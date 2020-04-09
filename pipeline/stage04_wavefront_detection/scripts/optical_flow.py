import numpy as np
from copy import copy
from scipy import interpolate

def anglesubtract(x, y, angleFlag=False):
    # ANGLESUBTRACT Subtracts the matrix y from the equally sized matrix x and
    # returns a value between -pi and +pi. This is valid if x and y both
    # contain angular data in radians. If angleFlag is false, returns a
    # non-angular subtraction.
    # Rory Townsend, Oct 2017
    # rory.townsend@sydney.edu.au

    if angleFlag:
        # METHOD 1: Modulo method
        xydiff = np.mod(x - y + np.pi, 2*np.pi) - np.pi
        # METHOD 2: Trig method, works but is slower
        # xydiff2 = atan2(sin(x-y), cos(x-y));
    else:
        xydiff = x - y
    return xydiff

def interpolate_nan_sites(frames, bad_sites=None, angle_flag=False):
    if frames.ndim < 2 or 3 < frames.ndim:
        raise InputError
    elif frames.ndim == 2:
        frames = frames[np.newaxis, :, :]
    else:
        pass

    # create union of bad sites and nan sites
    if bad_sites is None:
        bad_sites = np.array([[],[]])
    nan_sites = np.where(np.bitwise_not(np.isfinite(frames[0])))
    rm_idx = np.array([], dtype=int)
    for x,y in zip(bad_sites[0], bad_sites[1]):
        inx = np.where(x == nan_sites[0])[0]
        iny = np.where(y == nan_sites[1])[0]
        rm_idx = np.append(rm_idx,
                           np.intersect1d(inx, iny, assume_unique=True))
    nan_sites = (np.delete(nan_sites[0], rm_idx),
                 np.delete(nan_sites[1], rm_idx))
    nan_x, nan_y = np.append(nan_sites, bad_sites, axis=1)

    x, y = np.where(np.isfinite(frames[0]))
    nan_x, nan_y = np.where(np.bitwise_not(np.isfinite(frames[0])))

    # interpolate
    for i, frame in enumerate(frames):
        f = interpolate.interp2d(x, y, frame[x,y], kind='linear')
        znew = f(np.arange(frame.shape[0]),
                 np.arange(frame.shape[1]))
        frames[i][nan_x,nan_y] = znew.T[nan_x,nan_y]
    return frames

def phasegradient(f, badChannels=None, angleFlag=False):
    # % Calculates the phase gradient of complex matrix F
    # %
    # % Rory Townsend, Aug 2017
    # % rory.townsend@sydney.edu.au
    if f.ndim < 2 or 3 < f.ndim:
        raise InputError
    elif f.ndim == 2:
        f = f[np.newaxis, :, :]
    else:
        pass

    sf = f.shape

    # Convert phase data to complex coefficients
    if angleFlag and np.isreal(f).all():
        f = np.exp(1j*f);

    # Convert complex data to amplitudes
    if not angleFlag and not np.isreal(f).all():
        f = np.abs(f)


    # % Find channels with zero amplitude, as these will cause problems
    # if angleFlag
    #     badChannels = union(badChannels, find(any(abs(f)==0,3)));
    #     % Smooth out bad channels
    #     f = interpolateDeadElectrodes(f, badChannels);
    # end

    # Iteratively calculate gradient
    gfx = np.zeros_like(f)
    gfy = np.zeros_like(f)
    for i, frame in enumerate(f):
        if angleFlag:
            raise NotImplementedError
            # % Use modified MATLAB inbuilt function
            # [igfx, igfy] = phasegradientMATLAB(angle(f(:,:,it)), angleFlag);
            # gfx(:,:,it) = -igfx;
            # gfy(:,:,it) = -igfy;
        else:
            # Regular gradient
            gfx[i,:,:], gfy[i,:,:] = np.gradient(frame)

    return gfx, gfy #, badChannels


    # function [frow, fcol] = normalGrad(f, surroundLocs)
    # % Choose method to calculate regular, non-circular gradient
    # if ~isempty(surroundLocs)
    #     % If optional input SURROUNDLOCS is given, use the more accurate stencil
    #     % provided to calculate gradient (but may be slower)
    #     frow = reshape(surroundLocs.dy * f(:), size(f));
    #     fcol = reshape(surroundLocs.dx * f(:), size(f));
    #
    # else
    #     % Otherwise just use MATLAB's built-in gradient function
    #     [frow, fcol] = gradient(f);
    # end
    #
    # end


def optical_flow_step(frame, next_frame, nan_ids, surround_locs_dx,
                      surround_locs_dy, laplacian,
                      alpha, beta, u0, v0, prev_frame,
                      next2_frame, angle_flag):

    # Fixed point iteration parameters
    # Maximum fractional change between iterations to be counted as a fixed point
    maxChange = 0.01
    # Maximum number of iterations
    maxIter = 100
    # Starting relaxation parameter for fixed point iteration
    relaxParam = 1.1
    # Step to decrease relaxation parameter every iteration to ensure convergence
    relaxDecStep = 0.02
    # Minimum relaxation parameter
    relaxParamMin = 0.2

    u = u0 # copy?
    v = v0

    # Spatial derivatives
    Ex1, Ey1 = phasegradient(frame) #, nan_ids, angle_flag)
    Ex2, Ey2 = phasegradient(next_frame) #, nan_ids, angle_flag)
    Ex = (Ex1+Ex2)/2
    Ey = (Ey1+Ey2)/2
    # nan_ids = np.unique(np.concatenate(1, nan_ids, new_nan1, new_nan2))
    Ex[0][nan_ids] = 0
    Ey[0][nan_ids] = 0

    # Temporal derivative
    if not np.isreal(frame).all():
        Et = anglesubtract(np.angle(frame), np.angle(next_frame), angle_flag);
    else:
        if prev_frame is None or next2_frame is None:
            # Take centred difference
            Et = anglesubtract(next_frame, frame, angle_flag);
        else:
            # Use 5 point stencil
            Et = anglesubtract(1/12 * anglesubtract(prev_frame, next2_frame),
                               2/3 * anglesubtract(frame, next_frame), angle_flag)

    dataE = np.ones(frame.shape) * np.inf
    smoothE = np.ones(frame.shape) * np.inf

    # Loop over different non-linear penalty functions until a fixed point is reached
    convergence_loop = 0
    for loop in range(maxIter):

        lastDataE = copy(dataE) # copy?
        lastSmoothE = copy(smoothE)

        # Compute the first order error in data and smoothness
        dataE = np.multiply(Ex, u) + np.multiply(Ey, v) + Et
        upx, upy = phasegradient(u, nan_ids, 0)
        vpx, vpy = phasegradient(v, nan_ids, 0)
        smoothE = np.power(upx, 2) + np.power(upy, 2)\
                + np.power(vpx, 2) + np.power(vpy, 2)

        # Compute nonlinear penalty functions
        dataP   = 0.5/beta * np.power(beta**2 + np.power(dataE, 2), -1/2)
        smoothP = 0.5/beta * np.power(beta**2 + smoothE, -1/2)

        dataEChange = np.multiply(np.abs(dataE - lastDataE), 1/np.abs(dataE))
        # Check if data and smoothing errors have reached a fixed point
        smoothEChange = np.multiply(np.abs(smoothE - lastSmoothE), 1/np.abs(smoothE))

    # %    %TESTING ONLY: Show loop number and convergence properties at each step
    # %      totError = sum(abs(dataE(:)) + `alpha` * abs(sqrt(smoothE(:))));
    # %      disp([convergenceLoop, max(max(dataPChange)), max(max(smoothPChange)), ...
    # %          totError])

        # Exit loop if fixed point has been reached
        if np.max(dataEChange) < maxChange and np.max(smoothEChange) < maxChange:
            convergence_loop = loop
            break

        # Organize the discretized optical flow equations into a system of
        # linear equations in the form Ax=b.
        nrow, ncol = frame.shape
        N = nrow * ncol

        is_linear = False

        if is_linear:
            # Use original Horn-Schunk equations
            # gamma = 1 / alpha;
            gamma = dataP / alpha
            delta = 4 * smoothP
            surroundTerms = np.multiply(laplacian,
                                        smoothP[:, None])  # repmat(smoothP(:), 1, N))  ??
        else:
            # Use non-linear penalty function for more robust results (but
            # calculation may take more time)
            gamma = dataP / alpha
            delta = np.zeros_like(smoothP)

            # Surrounding terms are a combination of laplacian and first
            # spatial derivative terms
            psx, psy = phasegradient(smoothP, None, 0)

            np.multiply(psx.ravel(), np.ones((1, N)))
            np.multiply(surround_locs_dx, np.multiply(psx.ravel(), np.ones((1, N))))
            np.multiply(laplacian, np.multiply(smoothP.ravel(), np.ones((1, N))))

            surroundTerms = np.multiply(surround_locs_dx, np.multiply(psx.ravel(), np.ones((1, N))))\
                          + np.multiply(surround_locs_dy, np.multiply(psy.ravel(), np.ones((1, N))))\
                          + np.multiply(laplacian, np.multiply(smoothP.ravel(), np.ones((1, N))))

        # Calculate b vector
        b = np.concatenate((np.multiply(np.multiply(gamma.ravel(), Et.ravel()), Ex.ravel()),
                            np.multiply(np.multiply(gamma.ravel(), Et.ravel()), Ey.ravel())))
        # Add diagonal terms
        diag_vals = np.concatenate((delta.ravel()*-1 - np.multiply(np.power(Ex.ravel(), 2), gamma.ravel()),
                              delta.ravel()*-1 - np.multiply(np.power(Ey.ravel(), 2), gamma.ravel())))
        # Create sparse diagonal matrix
        A = np.diag(diag_vals)

        # Add off-diagonal terms for ui-vi dependence
        uvDiag = -1 * np.multiply(np.multiply(Ex.ravel(), Ey.ravel()), gamma.ravel())
        p_off_upp_diag = np.diag(uvDiag, k=N)
        p_off_low_diag = np.diag(uvDiag, k=-N)
        A = A + p_off_upp_diag + p_off_low_diag

        # Add other terms for surrounding locations
        A[:N, :N] += surroundTerms
        A[N:, N:] += surroundTerms

        # Solve this system of linear equations, adding a small value along the
        # diagonal to avoid potentially having a singular matrix
        eps = 1e-10
        diag_view = np.einsum('ii->i', A)
        diag_view += eps

        xexact = np.linalg.solve(A, b)

        # Reshape back to grids
        u = (1-relaxParam)*u + relaxParam * xexact[:N].reshape((nrow, ncol));
        v = (1-relaxParam)*v + relaxParam * xexact[-N:].reshape((nrow, ncol));

        # Gradually reduce the relaxation parameter to ensure the fixed point
        # iteration converges
        if relaxParam > relaxParamMin:
            relaxParam = relaxParam - relaxDecStep

    # %    % TESTING ONLY: Show optical flow field at each step
    # %    quiver(u,v)
    # %    drawnow
    return u, v, convergence_loop


def optical_flow(signal, nan_channels, alpha, beta, angle_flag):
#     Initialize result structures
    nframes, nrows, ncols = signal.shape
    ivxx = np.zeros((nrows, ncols))
    ivyy = np.zeros((nrows, ncols))
    velocityX = np.zeros((nframes-1, nrows, ncols))
    velocityY = np.zeros((nframes-1, nrows, ncols))
    all_conv_steps = np.zeros(nframes-1)
#     If data is not angular, normalize values by scaling by the overall mean
    if not angle_flag:
        signal = signal / np.nanmean(abs(signal), axis=0)

#     Interpolate over missing sites
    # signal = interpolate_nan_sites(signal, nan_channels)
    median_signal = np.median(signal)
    signal[np.where(np.bitwise_not(np.isfinite(signal)))] = median_signal

#     Initialize temporary variables
    prev_frame = None
    frame = signal[0,:,:]
    next_frame = signal[1,:,:]
    if nframes >= 3:
        next2_frame = signal[2,:,:]
    else:
        next2_frame = None

#     create souround locations for each sites
#     ...
    surround_locs_dx = np.zeros((nrows*ncols, nrows*ncols))
    surround_locs_dy = np.zeros((nrows*ncols, nrows*ncols))
    laplacian = np.zeros((nrows*ncols, nrows*ncols))

#     loop over all time steps
    for i, frame in enumerate(signal):
#         calculate optical flow
        [ivxx, ivyy, conv_steps] = optical_flow_step(frame=frame,
                                                   next_frame=next_frame,
                                                   nan_ids=nan_channels,
                                                   surround_locs_dx=surround_locs_dx,
                                                   surround_locs_dy=surround_locs_dy,
                                                   laplacian=laplacian,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   u0=ivxx,
                                                   v0=ivyy,
                                                   prev_frame=prev_frame,
                                                   next2_frame=next2_frame,
                                                   angle_flag=angle_flag)
#         Store results
        all_conv_steps[i] = conv_steps
        velocityX[i,:,:] = ivxx
        velocityY[i,:,:] = ivyy

#         Next set of frames
        prev_frame = copy(frame)
        frame = copy(next_frame)
        next_frame = copy(next2_frame)
        if i + 3 < nframes:
            next2_frame = signal[i+3, :, :]
        else:
            next2_frame = None

#      Display warning for number of steps that didn't converge
    if np.sum(all_conv_steps == 1000) > 0:
        print(f'Warning! {np.sum(all_conv_steps == 1000)} of {len(all_conv_steps)} time steps did not converge.')

    return velocityX, velocityY, all_conv_steps
