import numpy as np
import cv2
import pywt
from math import floor, log2, prod
import matplotlib.pyplot as plt

def DWToptimalBits(C, S, wName, R):
    """Compute optimal quantizer bits for DWT coefficients"""
    L = len(S) - 2  # number of DWT levels
    
    # Extract the detail and approximation DWT coefficients
    Coef = [None] * (3*L + 1)
    
    # For pywt, we need to handle coefficient structure differently
    # C is a list of arrays: [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
    j = 0
    for k in range(L):
        # Get coefficients for level k+1 (pywt stores from coarse to fine)
        level_idx = L - k  # pywt stores from last level to first
        if level_idx == L:
            # For the coarsest level, we have approximation and details
            if L == 1:
                cH, cV, cD = C[1]
                cA = C[0]
            else:
                cH, cV, cD = C[L]
                cA = C[0]
        else:
            if L == 1:
                cH, cV, cD = C[1]
                cA = C[0]
            else:
                cH, cV, cD = C[level_idx]
                cA = C[0]
        
        Coef[j] = cH
        Coef[j+1] = cV
        Coef[j+2] = cD
        j += 3
    
    # Approximation coefficients (always the first element in pywt)
    Coef[3*L] = C[0]
    
    # Compute and store the coefficient variances
    CoefVar = np.zeros(3*L + 1)
    for k in range(L):
        k1 = k * 3
        for j in range(3):
            if Coef[k1+j] is not None and Coef[k1+j].size > 0:
                CoefVar[k1+j] = np.std(Coef[k1+j]) ** 2
                if CoefVar[k1+j] == 0:
                    CoefVar[k1+j] = 1
    
    if Coef[3*L] is not None and Coef[3*L].size > 0:
        CoefVar[3*L] = np.std(Coef[3*L]) ** 2
    
    # Geometric mean of variances
    p = 1.0 / (3*L + 1)
    gm = 1.0
    
    for j in range(3*L + 1):
        if CoefVar[j] > 0:
            gm *= CoefVar[j]
    
    gm = gm ** p
    
    # Compute quantizer bits using coefficient variances and geometric mean
    Qbits = np.zeros(3*L + 1)
    for k in range(3*L + 1):
        if CoefVar[k] > 0:
            Qbits[k] = round(R + 0.5 * log2(CoefVar[k] / gm))
        if Qbits[k] < 0:
            Qbits[k] = 0
    
    # Compute the quantization steps
    Qsteps = np.zeros(3*L + 1)
    for k in range(3*L + 1):
        if Coef[k] is not None and Coef[k].size > 0:
            maxCoef = np.max(np.abs(Coef[k]))
            D = maxCoef
            if D != 0 and Qbits[k] > 0:
                Qsteps[k] = D / (2 * (2 ** Qbits[k]))
            else:
                Qsteps[k] = 1.0e+16
        else:
            Qsteps[k] = 1.0e+16
    
    return Qbits, Qsteps

def AssignIntgrBits2DWT(C, S, wName, R):
    """Assigns DWT coefficient quantizer bits optimally using recursive integer bit allocation rule"""
    L = len(S) - 2
    
    # Extract coefficients (similar to DWToptimalBits)
    Coef = [None] * (3*L + 1)
    j = 0
    for k in range(L):
        level_idx = L - k
        if level_idx == L:
            if L == 1:
                cH, cV, cD = C[1]
                cA = C[0]
            else:
                cH, cV, cD = C[L]
                cA = C[0]
        else:
            if L == 1:
                cH, cV, cD = C[1]
                cA = C[0]
            else:
                cH, cV, cD = C[level_idx]
                cA = C[0]
        
        Coef[j] = cH
        Coef[j+1] = cV
        Coef[j+2] = cD
        j += 3
    
    Coef[3*L] = C[0]
    
    # Compute variances
    CoefVar = np.zeros(3*L + 1)
    for k in range(L):
        k1 = k * 3
        for j in range(3):
            if Coef[k1+j] is not None and Coef[k1+j].size > 0:
                CoefVar[k1+j] = np.std(Coef[k1+j]) ** 2
                if CoefVar[k1+j] == 0:
                    CoefVar[k1+j] = 1
    
    if Coef[3*L] is not None and Coef[3*L].size > 0:
        CoefVar[3*L] = np.std(Coef[3*L]) ** 2
    
    Rtotal = (3*L + 1) * R  # total bits
    Qbits = np.zeros(3*L + 1)
    
    # Integer bit allocation
    while Rtotal > 0:
        max_val = -9999
        idx = 0
        for k in range(3*L + 1):
            if CoefVar[k] > max_val:
                max_val = CoefVar[k]
                idx = k
        
        Qbits[idx] += 1
        CoefVar[idx] = CoefVar[idx] / 2
        Rtotal -= 1
    
    # Compute quantization steps
    Qsteps = np.zeros(3*L + 1)
    for k in range(3*L + 1):
        if Coef[k] is not None and Coef[k].size > 0:
            maxCoef = np.max(np.abs(Coef[k]))
            D = maxCoef
            if D != 0 and Qbits[k] > 0:
                Qsteps[k] = D / (2 * (2 ** Qbits[k]))
            else:
                Qsteps[k] = 1.0e+16
        else:
            Qsteps[k] = 1.0e+16
    
    return Qbits, Qsteps

def quantizeDWT(C, S, Qsteps):
    """Quantize DWT coefficients"""
    L = len(S) - 2
    
    # In pywt, coefficients are stored as [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
    # We need to quantize in reverse order (from finest to coarsest)
    
    # Create a copy of coefficients to modify
    C_quantized = []
    for coef in C:
        if isinstance(coef, tuple):
            # Detail coefficients
            quantized_tuple = tuple(np.floor(c / Qsteps[i] + 0.5) * Qsteps[i] 
                                  for i, c in enumerate(coef))
            C_quantized.append(quantized_tuple)
        else:
            # Approximation coefficients (last in Qsteps array)
            C_quantized.append(np.floor(coef / Qsteps[-1] + 0.5) * Qsteps[-1])
    
    return C_quantized

def main():
    # Read image
    A = cv2.imread('yacht.ras')
    if A is None:
        print("Error: Could not read image 'yacht.ras'")
        return
    
    L = 3  # number of DWT levels
    
    # Make sure that the image size is divisible by 2^L
    height, width, depth = A.shape
    
    if height % (2**L) != 0:
        Height = floor(height / (2**L)) * (2**L)
    else:
        Height = height
        
    if width % (2**L) != 0:
        Width = floor(width / (2**L)) * (2**L)
    else:
        Width = width
        
    Depth = depth
    
    SamplingFormat = '4:2:2'  # for RGB image
    wName = 'db2'  # wavelet name
    
    # Bit budget: R1 for Y, R2 for Cb & R3 for Cr
    R1 = 1.0
    R2 = 0.5
    R3 = 0.5
    BitAssignRule = 'optimal'
    
    if Depth == 1:
        # Input image is B/W
        Y = A[0:Height, 0:Width].astype(np.float64) - 128
    else:
        # Input image is RGB. Convert it to YCbCr
        A1 = A[0:Height, 0:Width, :].astype(np.float64) - 128
        
        # Using standard YCbCr conversion coefficients
        Y = 0.299 * A1[:, :, 2] + 0.587 * A1[:, :, 1] + 0.114 * A1[:, :, 0]  # OpenCV uses BGR
        Cb = -0.168736 * A1[:, :, 2] - 0.331264 * A1[:, :, 1] + 0.5 * A1[:, :, 0]
        Cr = 0.5 * A1[:, :, 2] - 0.418688 * A1[:, :, 1] - 0.081312 * A1[:, :, 0]
        
        if SamplingFormat == '4:2:0':
            Cb = cv2.resize(Cb, (Width//2, Height//2), interpolation=cv2.INTER_CUBIC)
            Cr = cv2.resize(Cr, (Width//2, Height//2), interpolation=cv2.INTER_CUBIC)
        elif SamplingFormat == '4:2:2':
            Cb = cv2.resize(Cb, (Width//2, Height), interpolation=cv2.INTER_CUBIC)
            Cr = cv2.resize(Cr, (Width//2, Height), interpolation=cv2.INTER_CUBIC)
    
    # L-level 2D DWT of the Luma
    C_Y = pywt.wavedec2(Y, wName, level=L)
    
    # Get the coefficient structure for size information
    S = []
    S.append(C_Y[0].shape)  # Approximation coefficients
    for i in range(1, len(C_Y)):
        S.append(C_Y[i][0].shape)  # Detail coefficients
    
    # Compute optimal quantizer bits
    if BitAssignRule == 'optimal':
        Qbits, Qsteps = DWToptimalBits(C_Y, S, wName, R1)
    else:
        Qbits, Qsteps = AssignIntgrBits2DWT(C_Y, S, wName, R1)
    
    # Find total bits for the Y component
    TotalBitsY = (S[0][0] * S[0][1]) * np.sum(Qbits[0:3]) + \
                 (S[1][0] * S[1][1]) * np.sum(Qbits[3:6]) + \
                 (S[2][0] * S[2][1]) * np.sum(Qbits[6:9])
    
    print(f'Number of levels of DWT = {L}')
    print(f'Quantizer {BitAssignRule} bits')
    print('Quantizer bits:', Qbits)
    
    # Quantize and dequantize the coefficients
    Cq_Y = quantizeDWT(C_Y, S, Qsteps)
    Yq = pywt.waverec2(Cq_Y, wName)
    
    # Ensure the reconstructed image has the same size as original
    if Yq.shape != Y.shape:
        Yq = cv2.resize(Yq, (Width, Height))
    
    SNR = 20 * np.log10(np.std(Y) / np.std(Y - Yq))
    print(f'SNR(Y) = {SNR:.2f} dB')
    
    if Depth == 1:
        plt.figure()
        plt.imshow(Yq + 128, cmap='gray')
        AvgBit = TotalBitsY / (Height * Width)
        print(f'Desired avg. rate = {R1:.2f} bpp\tActual avg. rate = {AvgBit:.2f} bpp')
        plt.show()
    
    # If the input image is RGB, quantize the Cb & Cr components
    if Depth > 1:
        # Process Cb component
        C_Cb = pywt.wavedec2(Cb, wName, level=L)
        S_Cb = []
        S_Cb.append(C_Cb[0].shape)
        for i in range(1, len(C_Cb)):
            S_Cb.append(C_Cb[i][0].shape)
        
        if BitAssignRule == 'optimal':
            Qbits_Cb, Qsteps_Cb = DWToptimalBits(C_Cb, S_Cb, wName, R2)
        else:
            Qbits_Cb, Qsteps_Cb = AssignIntgrBits2DWT(C_Cb, S_Cb, wName, R2)
        
        TotalBitsCb = (S_Cb[0][0] * S_Cb[0][1]) * np.sum(Qbits_Cb[0:3]) + \
                      (S_Cb[1][0] * S_Cb[1][1]) * np.sum(Qbits_Cb[3:6]) + \
                      (S_Cb[2][0] * S_Cb[2][1]) * np.sum(Qbits_Cb[6:9])
        
        print('Cb quantizer bits:', Qbits_Cb)
        Cq_Cb = quantizeDWT(C_Cb, S_Cb, Qsteps_Cb)
        Cbq = pywt.waverec2(Cq_Cb, wName)
        
        if Cbq.shape != Cb.shape:
            Cbq = cv2.resize(Cbq, (Cb.shape[1], Cb.shape[0]))
        
        SNRcb = 20 * np.log10(np.std(Cb) / np.std(Cb - Cbq))
        
        # Process Cr component
        C_Cr = pywt.wavedec2(Cr, wName, level=L)
        S_Cr = []
        S_Cr.append(C_Cr[0].shape)
        for i in range(1, len(C_Cr)):
            S_Cr.append(C_Cr[i][0].shape)
        
        if BitAssignRule == 'optimal':
            Qbits_Cr, Qsteps_Cr = DWToptimalBits(C_Cr, S_Cr, wName, R3)
        else:
            Qbits_Cr, Qsteps_Cr = AssignIntgrBits2DWT(C_Cr, S_Cr, wName, R3)
        
        TotalBitsCr = (S_Cr[0][0] * S_Cr[0][1]) * np.sum(Qbits_Cr[0:3]) + \
                      (S_Cr[1][0] * S_Cr[1][1]) * np.sum(Qbits_Cr[3:6]) + \
                      (S_Cr[2][0] * S_Cr[2][1]) * np.sum(Qbits_Cr[6:9])
        
        print('Cr quantizer bits:', Qbits_Cr)
        
        # Find the overall average bit rate in bpp
        AvgBit = (TotalBitsY + TotalBitsCb + TotalBitsCr) / (Height * Width)
        print(f'Actual avg. rate = {AvgBit:.4f} bpp')
        
        Cq_Cr = quantizeDWT(C_Cr, S_Cr, Qsteps_Cr)
        Crq = pywt.waverec2(Cq_Cr, wName)
        
        if Crq.shape != Cr.shape:
            Crq = cv2.resize(Crq, (Cr.shape[1], Cr.shape[0]))
        
        SNRcr = 20 * np.log10(np.std(Cr) / np.std(Cr - Crq))
        
        # Upsample Cb & Cr to full resolution
        if SamplingFormat == '4:2:0':
            Cbq = cv2.resize(Cbq, (Width, Height), interpolation=cv2.INTER_CUBIC)
            Crq = cv2.resize(Crq, (Width, Height), interpolation=cv2.INTER_CUBIC)
        elif SamplingFormat == '4:2:2':
            Cbq = cv2.resize(Cbq, (Width, Height), interpolation=cv2.INTER_CUBIC)
            Crq = cv2.resize(Crq, (Width, Height), interpolation=cv2.INTER_CUBIC)
        
        # Do inverse component transformation & add DC level
        Ahat = np.zeros((Height, Width, Depth))
        Ahat[:, :, 2] = Yq + 1.402 * Crq + 128  # R
        Ahat[:, :, 1] = Yq - 0.344136 * Cbq - 0.714136 * Crq + 128  # G
        Ahat[:, :, 0] = Yq + 1.772 * Cbq + 128  # B
        
        # Clip values to valid range
        Ahat = np.clip(Ahat, 0, 255)
        
        plt.figure()
        plt.imshow(cv2.cvtColor(Ahat.astype(np.uint8), cv2.COLOR_BGR2RGB))
        print(f'Chroma sampling = {SamplingFormat}')
        print(f'SNR(Cb) = {SNRcb:.2f} dB\tSNR(Cr) = {SNRcr:.2f} dB')
        plt.show()

if __name__ == "__main__":
    main()
