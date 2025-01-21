%% 计算单张图的PSNR和SSIM
clear all
clc

Folder='..\Fig10\F-actin\';
ori_img=imread([Folder '/In focus.png']);
% ori_img=ori_img1(:,:,2);
% ori_img=imresize(ori_img,[512,512]);
res_img=imread([Folder '/out.png']);
% res_img=res_img1(:,:,2);
% res_img=imresize(res_img,[512,512]);

b_PSNR=psnr(res_img,ori_img);
b_SSIM=my_ssim(res_img,ori_img);
b_corr=corr2(res_img,ori_img);

%% 计算三张图的PSNR和SSIM
clear all
clc

Folder='..\Fig4\Fig4a\';
ori_img=imread([Folder '/In focus.png']);
res_img(:,:,1)=imresize(imread([Folder '/out_r.png']),[512,512]);
res_img(:,:,2)=imresize(imread([Folder '/out_g.png']),[512,512]);
res_img(:,:,3)=imresize(imread([Folder '/out_b.png']),[512,512]);

r_PSNR=psnr(res_img(:,:,1),ori_img(:,:,1));
r_SSIM=my_ssim(res_img(:,:,1),ori_img(:,:,1));
r_corr=corr2(res_img(:,:,1),ori_img(:,:,1));

g_PSNR=psnr(res_img(:,:,2),ori_img(:,:,2));
g_SSIM=my_ssim(res_img(:,:,2),ori_img(:,:,2));
g_corr=corr2(res_img(:,:,2),ori_img(:,:,2));

b_PSNR=psnr(res_img(:,:,3),ori_img(:,:,3));
b_SSIM=my_ssim(res_img(:,:,3),ori_img(:,:,3));
b_corr=corr2(res_img(:,:,3),ori_img(:,:,3));

a_PNSR=(r_PSNR+g_PSNR+b_PSNR)/3;
a_SSIM=(r_SSIM+g_SSIM+b_SSIM)/3;
a_corr=(r_corr+g_corr+b_corr)/3;

