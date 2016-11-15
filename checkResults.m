clc;
clear;
vectorize = input ('For checking squential results enter 0, for parallel enter 1\n');

check('conv1', 'Intermed_Results\2_conv1.mat', vectorize);
check('pool1', 'Intermed_Results\3_pool1.mat', vectorize);

check('fire2_squeeze1x1', 'Intermed_Results\4_fire2_squeeze1x1.mat', vectorize);
if (vectorize == 0)
    check('fire2_expand1x1', 'Intermed_Results\7_fire2_expand1x1.mat', vectorize);
    check('fire2_expand3x3', 'Intermed_Results\8_fire2_expand3x3.mat', vectorize);
end
check('fire2_concat', 'Intermed_Results\9_fire2_concat.mat', vectorize);

check('fire3_squeeze1x1', 'Intermed_Results\10_fire3_squeeze1x1.mat', vectorize);
if (vectorize == 0)
    check('fire3_expand1x1', 'Intermed_Results\13_fire3_expand1x1.mat', vectorize);
    check('fire3_expand3x3', 'Intermed_Results\14_fire3_expand3x3.mat', vectorize);
end
check('fire3_concat', 'Intermed_Results\15_fire3_concat.mat', vectorize);

check('fire4_squeeze1x1', 'Intermed_Results\16_fire4_squeeze1x1.mat', vectorize);
if (vectorize == 0)
    check('fire4_expand1x1', 'Intermed_Results\19_fire4_expand1x1.mat', vectorize);
    check('fire4_expand3x3', 'Intermed_Results\20_fire4_expand3x3.mat', vectorize);
end
check('fire4_concat', 'Intermed_Results\21_fire4_concat.mat', vectorize);

check('pool4', 'Intermed_Results\22_pool4.mat', vectorize);

check('fire5_squeeze1x1', 'Intermed_Results\23_fire5_squeeze1x1.mat', vectorize);
if (vectorize == 0)
    check('fire5_expand1x1', 'Intermed_Results\26_fire5_expand1x1.mat', vectorize);
    check('fire5_expand3x3', 'Intermed_Results\27_fire5_expand3x3.mat', vectorize);
end
check('fire5_concat', 'Intermed_Results\28_fire5_concat.mat', vectorize);

check('fire6_squeeze1x1', 'Intermed_Results\29_fire6_squeeze1x1.mat', vectorize);
if (vectorize == 0)
    check('fire6_expand1x1', 'Intermed_Results\32_fire6_expand1x1.mat', vectorize);
    check('fire6_expand3x3', 'Intermed_Results\33_fire6_expand3x3.mat', vectorize);
end
check('fire6_concat', 'Intermed_Results\34_fire6_concat.mat', vectorize);

check('fire7_squeeze1x1', 'Intermed_Results\35_fire7_squeeze1x1.mat', vectorize);
if (vectorize == 0)
    check('fire7_expand1x1', 'Intermed_Results\38_fire7_expand1x1.mat', vectorize);
    check('fire7_expand3x3', 'Intermed_Results\39_fire7_expand3x3.mat', vectorize);
end
check('fire7_concat', 'Intermed_Results\40_fire7_concat.mat', vectorize);

check('fire8_squeeze1x1', 'Intermed_Results\41_fire8_squeeze1x1.mat', vectorize);
if (vectorize == 0)
    check('fire8_expand1x1', 'Intermed_Results\44_fire8_expand1x1.mat', vectorize);
    check('fire8_expand3x3', 'Intermed_Results\45_fire8_expand3x3.mat', vectorize);
end
check('fire8_concat', 'Intermed_Results\46_fire8_concat.mat', vectorize);

check('pool8', 'Intermed_Results\47_pool8.mat', vectorize);

check('fire9_squeeze1x1', 'Intermed_Results\48_fire9_squeeze1x1.mat', vectorize);
if (vectorize == 0)
    check('fire9_expand1x1', 'Intermed_Results\51_fire9_expand1x1.mat', vectorize);
    check('fire9_expand3x3', 'Intermed_Results\52_fire9_expand3x3.mat', vectorize);
end
check('fire9_concat', 'Intermed_Results\53_fire9_concat.mat', vectorize);

check('conv10', 'Intermed_Results\54_conv10.mat', vectorize);

check('pool10', 'Intermed_Results\55_pool10.mat', vectorize);

check('prob', 'Intermed_Results\56_prob.mat', 0);