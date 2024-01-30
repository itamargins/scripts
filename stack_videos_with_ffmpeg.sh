ffmpeg \
-i /home/imagry/CenterNet/src/exp/ddd_asu/sheba_kia_COMBINED_itamar_100124_phase2/model_last_results/COMBINED_sheba_kia_100124_OVERSAMPLED_val/video_6.mp4 \
-i /home/imagry/CenterNet/src/exp/ddd_asu/ENTRON_zed_trucks_2023_12_12_phase2/COMBINED_sheba_kia_100124_OVERSAMPLED_val/video_6.mp4 \
-filter_complex "[0:v]drawtext=text='NEW model':x=10:y=10:fontsize=24:fontcolor=white[video1]; \
[1:v]drawtext=text='OLD model':x=10:y=10:fontsize=24:fontcolor=white[video2]; \
[video1][video2]vstack=inputs=2" \
-c:a copy /home/imagry/CenterNet/src/exp/ddd_asu/sheba_kia_COMBINED_itamar_100124_phase2/model_last_results/COMPARE_WITH_LATEST_MODEL/compare_6.mp4
