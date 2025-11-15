import cv2
import numpy as np
from math import floor, log2
import matplotlib.pyplot as plt

def jpgDCbits(dc, C):
    CodeLengthY = np.array([3, 4, 5, 5, 7, 8, 10, 12, 14, 16, 18, 20], dtype=np.int16)
    CodeLengthC = np.array([2, 3, 5, 6, 7, 9, 11, 13, 15, 18, 19, 20], dtype=np.int16)
    
    if C == 'Y':
        CodeLength = CodeLengthY
    else:
        CodeLength = CodeLengthC
        
    if dc == 0:
        return float(CodeLength[0])
    else:
        return float(CodeLength[round(log2(abs(dc)) + 0.5)])

def jpgACbits(x, C):
    RLCy = [None] * 16
    RLCy[0] = np.array([4, 3, 4, 6, 8, 10, 12, 14, 18, 25, 26], dtype=np.int16)
    RLCy[1] = np.array([5, 8, 10, 13, 16, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[2] = np.array([6, 10, 13, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[3] = np.array([7, 11, 14, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[4] = np.array([7, 12, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[5] = np.array([8, 12, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[6] = np.array([8, 13, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[7] = np.array([9, 13, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[8] = np.array([9, 17, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[9] = np.array([10, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[10] = np.array([10, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[11] = np.array([10, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[12] = np.array([11, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[13] = np.array([12, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[14] = np.array([13, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCy[15] = np.array([12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    
    RLCc = [None] * 16
    RLCc[0] = np.array([2, 3, 5, 7, 9, 12, 15, 23, 24, 25, 26], dtype=np.int16)
    RLCc[1] = np.array([5, 8, 11, 14, 20, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[2] = np.array([6, 9, 13, 15, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[3] = np.array([6, 9, 12, 15, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[4] = np.array([6, 10, 14, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[5] = np.array([7, 11, 17, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[6] = np.array([8, 12, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[7] = np.array([7, 12, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[8] = np.array([8, 14, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[9] = np.array([9, 14, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[10] = np.array([9, 14, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[11] = np.array([9, 14, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[12] = np.array([9, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[13] = np.array([11, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[14] = np.array([13, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    RLCc[15] = np.array([11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int16)
    
    if C == 'Y':
        RLC = RLCy
    else:
        RLC = RLCc
        
    x1 = ZigZag(x)
    
    k = 1  # Python uses 0-indexing, adjusted from MATLAB's 1-indexing
    Count = 0
    Bits = 0.0
    
    while k < 64:  # 0-63 for 64 elements
        if x1[k] == 0:
            Count += 1
            if k == 63:  # Last element
                Bits += float(RLC[0][0])
                break
        else:
            if Count == 0:
                RL = Count
                Level = round(log2(abs(x1[k])) + 0.5)
                Bits += float(RLC[RL][Level])
            elif 1 <= Count <= 15:
                RL = Count
                Level = round(log2(abs(x1[k])) + 0.5)
                Bits += float(RLC[RL][Level - 1])  # Adjusted indexing
                Count = 0
            else:
                Bits += float(RLC[15][0])
                Count -= 16
        k += 1
        
    return Bits

def ZigZag(x):
    # Returns the zigzag scan addresses of an 8x8 array
    y = np.zeros(64)
    y[0] = x[0, 0]; y[1] = x[0, 1]; y[2] = x[1, 0]; y[3] = x[2, 0]
    y[4] = x[1, 1]; y[5] = x[0, 2]; y[6] = x[0, 3]; y[7] = x[1, 2]
    y[8] = x[2, 1]; y[9] = x[3, 0]; y[10] = x[4, 0]; y[11] = x[3, 1]
    y[12] = x[2, 2]; y[13] = x[1, 3]; y[14] = x[0, 4]; y[15] = x[0, 5]
    y[16] = x[1, 4]; y[17] = x[2, 3]; y[18] = x[3, 2]; y[19] = x[4, 1]
    y[20] = x[5, 0]; y[21] = x[6, 0]; y[22] = x[5, 1]; y[23] = x[4, 2]
    y[24] = x[3, 3]; y[25] = x[2, 4]; y[26] = x[1, 5]; y[27] = x[0, 6]
    y[28] = x[0, 7]; y[29] = x[1, 6]; y[30] = x[2, 5]; y[31] = x[3, 4]
    y[32] = x[4, 3]; y[33] = x[5, 2]; y[34] = x[6, 1]; y[35] = x[7, 0]
    y[36] = x[7, 1]; y[37] = x[6, 2]; y[38] = x[5, 3]; y[39] = x[4, 4]
    y[40] = x[3, 5]; y[41] = x[2, 6]; y[42] = x[1, 7]; y[43] = x[2, 7]
    y[44] = x[3, 6]; y[45] = x[4, 5]; y[46] = x[5, 4]; y[47] = x[6, 3]
    y[48] = x[7, 2]; y[49] = x[7, 3]; y[50] = x[6, 4]; y[51] = x[5, 5]
    y[52] = x[4, 6]; y[53] = x[3, 7]; y[54] = x[4, 7]; y[55] = x[5, 6]
    y[56] = x[6, 5]; y[57] = x[7, 4]; y[58] = x[7, 5]; y[59] = x[6, 6]
    y[60] = x[5, 7]; y[61] = x[6, 7]; y[62] = x[7, 6]; y[63] = x[7, 7]
    return y

def main():
    # Read image
    A = cv2.imread('birds.ras')
    if A is None:
        print("Error: Could not read image 'birds.ras'")
        return
    
    Height, Width, Depth = A.shape
    N = 8  # Transform matrix size
    
    # Limit Height & Width to multiples of 8
    if Height % N != 0:
        Height = floor(Height / N) * N
    if Width % N != 0:
        Width = floor(Width / N) * N
        
    A1 = A[0:Height, 0:Width, :]
    A = A1.copy()
    
    SamplingFormat = '4:2:0'
    
    if Depth == 1:
        y = A.astype(np.float64)
    else:
        A_ycbcr = cv2.cvtColor(A, cv2.COLOR_BGR2YCrCb)
        A = A_ycbcr.astype(np.float64)
        y = A[:, :, 0]
        
        if SamplingFormat == '4:2:0':
            Cb = cv2.resize(A[:, :, 1], (Width // 2, Height // 2), interpolation=cv2.INTER_CUBIC)
            Cr = cv2.resize(A[:, :, 2], (Width // 2, Height // 2), interpolation=cv2.INTER_CUBIC)
        elif SamplingFormat == '4:2:2':
            Cb = cv2.resize(A[:, :, 1], (Width // 2, Height), interpolation=cv2.INTER_CUBIC)
            Cr = cv2.resize(A[:, :, 2], (Width // 2, Height), interpolation=cv2.INTER_CUBIC)
    
    # Quantization tables
    jpgQstepsY = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    QstepsY = jpgQstepsY.copy()
    Qscale = 1.5
    
    Yy = np.zeros((N, N))
    xqY = np.zeros((Height, Width))
    acBitsY = 0
    dcBitsY = 0
    
    if Depth > 1:
        jpgQstepsC = np.array([
            [17, 18, 24, 47, 66, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ])
        
        QstepsC = jpgQstepsC.copy()
        YCb = np.zeros((N, N))
        YCr = np.zeros((N, N))
        
        if SamplingFormat == '4:2:0':
            xqCb = np.zeros((Height // 2, Width // 2))
            xqCr = np.zeros((Height // 2, Width // 2))
        elif SamplingFormat == '4:2:2':
            xqCb = np.zeros((Height, Width // 2))
            xqCr = np.zeros((Height, Width // 2))
            
        acBitsCb = 0
        dcBitsCb = 0
        acBitsCr = 0
        dcBitsCr = 0
    
    # Compute the bits for the Y component
    for m in range(0, Height, N):
        for n in range(0, Width, N):
            t = y[m:m+N, n:n+N] - 128
            Yy = cv2.dct(t)  # N x N 2D DCT of input image
            
            # quantize the DCT coefficients
            temp = np.floor(Yy / (Qscale * QstepsY) + 0.5)
            
            # Calculate bits for the DC difference
            if n == 0:
                DC = temp[0, 0]
                dcBitsY += jpgDCbits(DC, 'Y')
            else:
                DC_diff = temp[0, 0] - DC
                dcBitsY += jpgDCbits(DC_diff, 'Y')
                DC = temp[0, 0]
            
            # Calculate the bits for the AC coefficients
            ACblkBits = jpgACbits(temp, 'Y')
            acBitsY += ACblkBits
            
            # dequantize & IDCT the DCT coefficients
            temp_dequant = temp * (Qscale * QstepsY)
            xqY[m:m+N, n:n+N] = cv2.idct(temp_dequant) + 128
    
    # If the input image is a color image, calculate the bits for the chroma components
    if Depth > 1:
        if SamplingFormat == '4:2:0':
            EndRow = Height // 2
        else:
            EndRow = Height
            
        for m in range(0, EndRow, N):
            for n in range(0, Width // 2, N):
                t1 = Cb[m:m+N, n:n+N] - 128
                t2 = Cr[m:m+N, n:n+N] - 128
                
                Ycb = cv2.dct(t1)  # N x N 2D DCT of Cb image
                Ycr = cv2.dct(t2)
                
                temp1 = np.floor(Ycb / (Qscale * QstepsC) + 0.5)
                temp2 = np.floor(Ycr / (Qscale * QstepsC) + 0.5)
                
                if n == 0:
                    DC1 = temp1[0, 0]
                    DC2 = temp2[0, 0]
                    dcBitsCb += jpgDCbits(DC1, 'C')
                    dcBitsCr += jpgDCbits(DC2, 'C')
                else:
                    DC1_diff = temp1[0, 0] - DC1
                    DC2_diff = temp2[0, 0] - DC2
                    dcBitsCb += jpgDCbits(DC1_diff, 'C')
                    dcBitsCr += jpgDCbits(DC2_diff, 'C')
                    DC1 = temp1[0, 0]
                    DC2 = temp2[0, 0]
                
                ACblkBits1 = jpgACbits(temp1, 'C')
                ACblkBits2 = jpgACbits(temp2, 'C')
                acBitsCb += ACblkBits1
                acBitsCr += ACblkBits2
                
                # dequantize and IDCT the coefficients
                temp1_dequant = temp1 * (Qscale * QstepsC)
                temp2_dequant = temp2 * (Qscale * QstepsC)
                xqCb[m:m+N, n:n+N] = cv2.idct(temp1_dequant) + 128
                xqCr[m:m+N, n:n+N] = cv2.idct(temp2_dequant) + 128
    
    # Calculate SNR and display results
    mse = np.std(y - xqY)
    snr = 20 * np.log10(np.std(y) / mse)
    print(f'SNR = {snr:.2f}')
    
    if Depth == 1:
        TotalBits = acBitsY + dcBitsY
        plt.figure()
        plt.imshow(xqY, cmap='gray')
        plt.title(f'JPG compressed @ {TotalBits/(Height*Width):.2f} bpp')
        plt.show()
    else:
        TotalBits = acBitsY + dcBitsY + dcBitsCb + acBitsCb + dcBitsCr + acBitsCr
        
        c1 = cv2.resize(xqCb, (Width, Height), interpolation=cv2.INTER_CUBIC)
        c2 = cv2.resize(xqCr, (Width, Height), interpolation=cv2.INTER_CUBIC)
        
        mseb = np.std(A[:, :, 1] - c1)
        snrb = 20 * np.log10(np.std(A[:, :, 1]) / mseb)
        msec = np.std(A[:, :, 2] - c2)
        snrc = 20 * np.log10(np.std(A[:, :, 2]) / msec)
        
        print(f'SNR(Cb) = {snrb:.2f}dB\tSNR(Cr) = {snrc:.2f}dB')
        
        xq = np.zeros((Height, Width, 3))
        xq[:, :, 0] = xqY
        xq[:, :, 1] = c1
        xq[:, :, 2] = c2
        
        xq_rgb = cv2.cvtColor(np.uint8(np.round(xq)), cv2.COLOR_YCrCb2BGR)
        
        plt.figure()
        plt.imshow(cv2.cvtColor(xq_rgb, cv2.COLOR_BGR2RGB))
        plt.title(f'JPG compressed @ {TotalBits/(Height*Width):.2f} bpp')
        plt.show()
    
    print(f'Bit rate = {TotalBits/(Height*Width):.2f} bpp')

if __name__ == "__main__":
    main()
