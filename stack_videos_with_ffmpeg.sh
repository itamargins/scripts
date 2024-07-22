ffmpeg \
-i /home/itamar/mnt/ViT_vs_CN/CN_ZED/evaluation/3DASU_2619bfb/192_3d_asu_val/eval_optim_thresh/video_1.mp4 \
-i /home/itamar/mnt/ViT_vs_CN/COMPARISONS/ViT_upToDate_VS_CN/ViT_upToDate/eval_optim_thresh/video_1.mp4 \
-filter_complex "[0:v]drawtext=text='CenterNet':x=10:y=10:fontsize=72:fontcolor=white[video1]; \
[1:v]drawtext=text='EfficientViT_upToDate':x=10:y=10:fontsize=72:fontcolor=white[video2]; \
[video1][video2]hstack=inputs=2" \
-c:a copy /home/itamar/mnt/ViT_vs_CN/COMPARISONS/ViT_upToDate_VS_CN/compare_video_1.mp4