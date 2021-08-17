function [imin,yblk, F0, F0m] = align_block2_nonrigid_target(F, ysamp, nblocks)

threshold1 = 0.1;
threshold2 = 0.15;

fprintf('Align block with non-rigid target template cor output\n')

% F is y bins by amp bins by batches
% ysamp are the coordinates of the y bins in um

Nbatches = size(F,3);

% look up and down this many y bins to find best alignment
n = 30;
dc = zeros(2*n+1, Nbatches);
dt = -n:n;

% we do everything on the GPU for speed, but it's probably fast enough on
% the CPU
Fg = gpuArray(single(F));

% mean subtraction to compute covariance
Fg = Fg - mean(Fg, 1);

% initialize the target "frame" for alignment with a single sample
F0 = Fg(:,:, min(300, floor(size(Fg,3)/2)));

% We do rigid registration by integer shifts,
% Then upsample and do non-rigid registration (register each block)
% The target frame is updated at each iteration (both rigid and non rigid)
niter_rigid = 10;
niter_nonrigid = 10;

% figure out how to split the probe into nblocks pieces
% if nblocks = 1, then we're doing rigid registration
nybins = size(F,1);
yl = floor(nybins/nblocks)-1;
ifirst = round(linspace(1, nybins - yl, 2*nblocks-1));
ilast  = ifirst + yl; %287;
nblocks = length(ifirst);
yblk = zeros(length(ifirst), 1);

% Keep track of the shifts at each iteration
dall = zeros(niter_rigid + niter_nonrigid, Nbatches, nblocks);

%% first we do rigid registration by integer shifts
% everything is iteratively aligned until most of the shifts become 0.

for iter = 1:niter_rigid
    for t = 1:length(dt)
        % for each NEW potential shift, estimate covariance
        Fs = circshift(Fg, dt(t), 1);
        dc(t, :) = gather(sq(mean(mean(Fs .* F0, 1), 2)));
    end
    
    % estimate the best shifts
    [~, imax] = max(dc, [], 1);

    % align the data by these integer shifts
    for t = 1:length(dt)
        ib = imax==t;
        Fg(:, :, ib) = circshift(Fg(:, :, ib), dt(t), 1);
        for j = 1:nblocks
            % Same shift for each block during this rigid main loop
            dall(iter, ib, j) = dt(t);
        end
    end
    
    % new target frame based on our current best alignment
    F0 = mean(Fg, 3);
end

%%

% for each small block, we only look up and down this many samples to find
% nonrigid shift
n = 15;  % TODO (TB): Increase back to 30 and check if it works with Valentino's high non-rigid drift chronic data
dt = -n:n;
dcs = zeros(2*n+1, Nbatches, nblocks);

%% Non-rigid iterations (integer shifts)
% Now we register each block and update the target frame
% All shifts are integer except the very last iteration

% We dismiss blocks/batches with presumably artifactual drift estimation
% when computing the target fingerprint (so that it's not polluted)
good_indexes_logical = zeros(nblocks,Nbatches);

% Calculate variances used to compute correlation, since they
% aren't supposed to depend on the shift (actually is not a true variance but ok)
dev1 = zeros(nblocks,Nbatches); % later i will pre-allocate for speed
dev2 = zeros(nblocks,Nbatches);
for j=1:nblocks
    isub = ifirst(j):ilast(j);
    dev1(j,:) = gather(sqrt(sq(mean(mean(Fg(isub, :, :) .* Fg(isub, :, :), 1), 2))));
    dev2(j,:) = gather(sqrt(sq(mean(mean(F0(isub, :, :) .* F0(isub, :, :), 1), 2))));  % TODO (TB): I think this one should in theory be recomputed at each iteration
end
devtot = dev1.*dev2;

% integer shift for all iterations except last one
for iter = 1:niter_nonrigid - 1
    
    % Shift each block independently
    for j = 1:nblocks
        isub = ifirst(j):ilast(j);
        yblk(j) = mean(ysamp(isub));
        
        Fsub = Fg(isub, :, :);
        
        for t = 1:length(dt)
            % for each potential shift, estimate CORRELATION (rather than covariance)
            % We pick correlation so that the threshold used to filter-out bad batches/blocks
            % doesn't depend on batch/block size and amplitude
            Fs = circshift(Fsub, dt(t), 1);
            dcs(t, :, j) = gather(sq(mean(mean(Fs .* F0(isub, :, :), 1), 2)));
            dcs(t, :, j) = dcs(t, :, j)./devtot(j,:);
        end

        % estimate the best shifts
        [cor_max, imax] = max(dcs(:, :, j), [], 1);
        
        if iter < niter_nonrigid -2
            % Up to the ~end, average over all batches/blocks to update target fingerprint
            good_indexes1 = 1:Nbatches;
        else
            % Towards the end, average only batches/blocks
            % with above threshold-correlation to update target fingerprint
            % (So we hopefully remove some noise in the template before the last iteration)
            % (NB: we filter out bad batches/blocks at the end and not the beginning
            % because correlations to the template are higher then)
            [cor_min, ~] = min(dcs(:, :, j), [], 1);
            cor_diff = cor_max - cor_min;
            good_indexes1 = find(cor_diff > threshold1);
            logicals = cor_diff > threshold1;
        end


        % align the data by these integer shifts
        for t = 1:length(dt)
            ib = imax==t;
            % NB (TB): circular shift could become problematic for large shifts/small blocks
            % I believe 0-padding would be better in theory
            Fg(isub, :, ib) = circshift(Fg(isub, :, ib), dt(t), 1);
            dall(niter_rigid + iter, ib, j) = dt(t);
        end
        
        % new target frame based on current best alignment for current block
        % can I do logical indexing? If yes, we can speed up
        F0(isub, :) = mean(Fg(isub, :, good_indexes1), 3);
        
        if iter > niter_nonrigid -2
            good_indexes_logical(j,:) = logicals;     
        end
        
    end
end

%% (last iteration): sub-integer shifts

% to find sub-integer shifts for each block ,
% we now use upsampling, based on kriging interpolation
dtup = linspace(-n, n, (2*n*10)+1);
K = kernelD(dt,dtup,1); % this kernel is fixed as a variance of 1
dcs = my_conv2(dcs, .5, [1, 2, 3]); % some additional smoothing for robustness, across all dimensions

for j = 1:nblocks
    % using the upsampling kernel K, get the upsampled cross-correlation
    % curves
    dcup = K' * dcs(:,:,j);
    
    % find the  max of these curves
    [cor_max, imax] = max(dcup, [], 1);
    [cor_min, ~] = min(dcup, [], 1);
    
    cor_diff = cor_max - cor_min;
    good_indexes2 = find(cor_diff > threshold2);
    
    % add the value of the shift to the last row of the matrix of shifts
    % (as if it was the last iteration of the main non-rigid loop )
    dall(niter_rigid + niter_nonrigid, :, j) = dtup(imax);
    
end


% the sum of all the shifts (rigid and non-rigid) equals the final shifts for each block
imin = zeros(Nbatches, nblocks);

% For all the filtered-out batches/blocks, final drift is that of the previous batch
% (TODO: Ideally we should interpolate)
for j = 1:nblocks
    imin(1,j) = sum(dall(:,1,j),1);
    for k = 2:Nbatches
        if good_indexes_logical(j,k)
            imin(k,j) = sum(dall(:,k,j),1);
        else
            imin(k,j) = imin(k-1,j);
        end
    end
end


%%

% Doesn't include the last upsampled iteration
F0m = mean(Fg(:, :, good_indexes2),3);

end