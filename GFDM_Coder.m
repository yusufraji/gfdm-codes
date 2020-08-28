
%TU Dresden GFDM Modulator Code Adaptation...
function [GFDM_Tx_real, GFDM_Tx_imag] = GFDM_Coder(K, M, Kon, ...
    BitRateDefault, SampleRateDefault, TimeWindow, mu, numBlocks, ...
    GFDMTxReal, GFDMTxImag, GFDMRowSize, GFDMInpData, GFDMComplex, ...
    Pstruct, ZerosAdded, BC, TransmitQAM, ChannelLabel, UpsampleRate);
%
% clear all;
% clc;
% K = 128;
% M = 7;
% Kon = 128;
% mu = 2;
% numBlocks = 1;
% BitRateDefault = 2.5*(10^9);
% SampleRateDefault = 4*BitRateDefault;
% TimeWindow = 8*128/BitRateDefault;

p = struct;

p.K = K;             %   K:              Number of samples per sub-symbol (default = 1024)
p.Kon = Kon;           %   Kon:            Number allocated subcarriers (default = 600)
p.M = M;                %   M:              Number of sub-symbols (Default = 15)
p.Ncp = 0;              %   Ncp:            Number of cyclic prefix samples
p.Ncs = 0;              %   Ncs:            Number of cyclic suffix samples
p.window = 'rc';        %   window:         window that multiplies the GFDM block with CP
p.b = 0;                %   b:              window factor (rolloff of the window in samples)
p.overlap_blocks = 0 ;  %   overlap_blocks: overlaped blocks (number of overlaped edge samples)
p.matched_window = 0 ;  %   matched_window: matched windowing (root raised windowing in AWGN)
% % Always use a RRC filter with rolloff 0.5 (if not otherwise stated)
p.pulse = 'rrc';        %   pulse:          Pulse shaping filter
p.sigmaN = 0;           %   sigmaN:         noise variance information for MMSE receiver
p.a = 0.18;              %   a:              rolloff of the pulse shaping filter
p.L = 2;                %   L:              Number of overlapping subcarriers
p.mu = mu;               %   mu:             Modulation order (number of bits in the QAM symbol)
p.oQAM = 0;             %   oQAM:           offset QAM Modulation/Demodulation
p.B = numBlocks;                %   B:              number of concatenated GFDM blocks
% p=get_defaultGFDM('TTI');

%S'ha d'anar amb compte, sino depen de quina configuracio de p agafis et
%dona error el zero forcing
%You have to be careful, but depend on which p config you take gives you
%the zero forcing error

% the sample rate has to give a power of two number of samples per bit
if ~isequal(ceil(log2(SampleRateDefault/BitRateDefault)),floor(log2(SampleRateDefault/BitRateDefault)))
    SR = num2str(SampleRateDefault);
    SPB = num2str(SampleRateDefault/BitRateDefault);
    error(['SampleRateDefault ' SR ' does not yield a number of samples per bit which is a power of two, but ' SPB ' samples per bit']);
end;

N = p.K*p.M;
Truncate = N;   % This is returned to VPI to chop off the bitstream if zeros are added
L = p.Kon*p.M;
numBlocks = p.B;
GFDM_Complex = complex(zeros(N,numBlocks));
B_C = complex(zeros(L,numBlocks));
inpData = zeros(L, numBlocks);
% seed the random number generator
rng(42)
for block = 1:numBlocks
    % create symbols
    inpData(:,block) = get_random_symbols(p);
    
    % map them to qam and to the D matrix
    B_C(:,block) = do_qammodulate(inpData(:,block), p.mu);
    D = do_map(p, B_C(:,block)); %Les dimensions de la matriu seran NK,CC
    GFDM_Complex(:,block) = do_modulate(p, D);
end
% normalize the constallation. to be used at the receiver
A = zeros(L,numBlocks);
A_real = zeros(L,numBlocks);
A_imag = zeros(L,numBlocks);
New_B_C = zeros(L,numBlocks);
for block = 1:numBlocks
    % Normalisation for constellation plot
    A(:,block) = sqrt(real(B_C(:,block)).^2 + imag(B_C(:,block)).^2);
    Max_A = max(A(:,block));
    A_real(:,block) = real(B_C(:,block))./Max_A;
    A_imag(:,block) = imag(B_C(:,block))./Max_A;
    New_B_C(:,block) = A_real(:,block) + 1i*A_imag(:,block);
end
% B_C = New_B_C;
% end of normalisation

GFDM_real = zeros(N,2*numBlocks);

GFDM_Mod = zeros(N,numBlocks);
GFDM_Mod_real = zeros(N,numBlocks);
GFDM_Mod_imag = zeros(N,numBlocks);
GFDM_Tx = zeros(N,numBlocks);
GFDM_Tx_real = zeros(N,numBlocks);
GFDM_Tx_imag = zeros(N,numBlocks);

for block = 1:numBlocks
    % Normalisation
    GFDM_Mod(:,block) = sqrt(real(GFDM_Complex(:,block)).^2 + imag(GFDM_Complex(:,block)).^2);
    Max_GFDM_Mod = max(GFDM_Mod(:,block));
    GFDM_Mod_real(:,block) = real(GFDM_Complex(:,block))./Max_GFDM_Mod;
    GFDM_Mod_imag(:,block) = imag(GFDM_Complex(:,block))./Max_GFDM_Mod;
    GFDM_Tx(:,block) = GFDM_Mod_real(:,block) + 1i*GFDM_Mod_imag(:,block);
    
    GFDM_Tx_real(:,block) = real(GFDM_Tx(:,block));
    GFDM_Tx_imag(:,block) = imag(GFDM_Tx(:,block));
    
    % this GFDM_real will be used to plot here in matlab during debugging
    GFDM_real(:,(block*2)) = imag(GFDM_Complex(:,block));
    GFDM_real(:,(block*2)-1) = real(GFDM_Complex(:,block));
end

%Paralel to serial
GFDM_real = reshape(GFDM_real, N*2*numBlocks, 1);

GFDM_Tx_real = reshape(GFDM_Tx_real, N*numBlocks, 1);
GFDM_Tx_imag = reshape(GFDM_Tx_imag, N*numBlocks, 1);
[rowSize colSize] = size(GFDM_Tx_real);

% ensure that the size is a power of 2
if ~isequal(floor(log2(rowSize)), ceil(log2(rowSize)))
    
    % add zeros to make the number of bits a power of 2 and place the data in
    % the middle
    Zeros_Added = 2^(nextpow2(rowSize))-rowSize;
    GFDM_Tx_real = [zeros(1,Zeros_Added/2),GFDM_Tx_real(:)',zeros(1,Zeros_Added/2)]';
    GFDM_Tx_imag = [zeros(1,Zeros_Added/2),GFDM_Tx_imag(:)',zeros(1,Zeros_Added/2)]';
    
end
% UPSAMPLING: ALWAYS UPSAMPLE BY AN EVEN NUMBER TO MAINTAIN THE POWER OF 2
GFDM_Tx_real = upsample(GFDM_Tx_real, UpsampleRate);
GFDM_Tx_imag = upsample(GFDM_Tx_imag, UpsampleRate);

[rowSize colSize] = size(GFDM_Tx_real);
%% Plot the resulting PSD
f = linspace(-p.K/2, p.K/2, 2*length(GFDM_real)+1); f = f(1:end-1)';
g = mag2db(fftshift(abs(fft(GFDM_real, 2*length(GFDM_real)))))/2;

% save ('C:\Users\PON Simulator\Documents\Yusuf\Debug_GFDM_Coder.mat')
% save ('Debug_GFDM_Coder.mat')
[filepath,name,ext] = fileparts(GFDMTxReal);
disp(filepath);
mkdir(filepath, ChannelLabel);
fullFileName = fullfile(filepath, ChannelLabel, 'GFDMTxReal.mat');
save(fullFileName,'GFDM_Tx_real');
fullFileName = fullfile(filepath, ChannelLabel, 'GFDMTxImag.mat');
save (fullFileName,'GFDM_Tx_imag');
fullFileName = fullfile(filepath, ChannelLabel, 'GFDMRowSize.txt');
dlmwrite (fullFileName,rowSize, 'precision', 10);

% dlmwrite (TransmitQAM,B_C, 'precision', 10)
fullFileName = fullfile(filepath, ChannelLabel, 'TransmitQAM.mat');
save (fullFileName, 'B_C');
fullFileName = fullfile(filepath, ChannelLabel, 'ZerosAdded.mat');
save (fullFileName,'Zeros_Added');
fullFileName = fullfile(filepath, ChannelLabel, 'GFDMInputData.mat');
save (fullFileName,'inpData');
fullFileName = fullfile(filepath, ChannelLabel, 'GFDMComplex.mat');
save (fullFileName,'GFDM_Complex');
fullFileName = fullfile(filepath, ChannelLabel, 'BC.mat');
save (fullFileName,'B_C');
fullFileName = fullfile(filepath, ChannelLabel, 'Pstruct.mat');
save (fullFileName,'p');
% figure();
% plot(f, g, 'r'); hold off;
% ylim([-40, 40]);
% xlabel('f/F'); ylabel('PSD [dB]');
% grid()
% legend({'GFDM'});
end
