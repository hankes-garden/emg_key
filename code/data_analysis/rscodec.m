function ret = rscodec(data, enc, n, k, m)
    if(enc) % encode
        msg_bin = data;
        msg_dec = toRSDecimal(msg_bin, k, m);
        msg_gf = gf(msg_dec, m);
        code = rsenc(msg_gf,n, k);
        code_bin = toRSBinary(code.x, m);
        ret = code_bin;
    else % decode
        code_bin = data;
        code_dec = toRSDecimal(code_bin, n, m);
        code_gf = gf(code_dec, m);
        decoded_dec = rsdec(code_gf, n, k);
        decoded_bin = toRSBinary(decoded_dec.x, m);
        ret = decoded_bin;
    end
return

function ret = toRSDecimal(data, col, m)
% transform a binary matrix to decimal matrix, in which each m bits represent 
% a decimal value
ret = [];
    for i = 0:col-1
        deCol = bi2de(data(:, i*m+1:(i+1)*m), 'left-msb');
        ret = [ret, deCol];
    end
return

function ret = toRSBinary(data, m)
% transform a decimal matrix to a binary matrix
    ret = [];
    for i = 1: size(data, 2)
        deCol = data(:, i);
        biMat = de2bi(deCol, 'left-msb', m);
        ret = [ret, biMat];
    end
return
