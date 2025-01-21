%% 计算多张图的PSNR和SSIM
clear all
clc
Folder='..\Fig12\F-actin\';
list=dir(strcat(Folder, '\GT\*.png'));      %水平串联字符串
for k1=1:max(size(list))
    name{k1}=getfield(list,{k1,1},'name');%把图片名存入元胞数组name
end

for i=1:k1
    ori_img=imread([Folder '\GT\' num2str(name{i})]);
    res_img=imread([Folder '\out\' num2str(name{i})]);
    b_PSNR(i,1)=psnr(res_img,ori_img);
    b_SSIM(i,1)=my_ssim(res_img,ori_img);
    b_corr(i,1)=corr2(res_img,ori_img);
    i
end

a_PSNR=mean(b_PSNR);
a_SSIM=mean(b_SSIM);
a_corr=mean(b_corr);