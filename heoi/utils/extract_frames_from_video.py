import os
import threading
import pdb


NUM_THREADS = 10
VIDEO_ROOT = '/home/liuzyng/Desktop/CVPR2019/data/something/video'         # Downloaded webm videos
FRAME_ROOT = '/home/liuzyng/Desktop/CVPR2019/data/something/frame'  # Directory for extracted frames


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def extract(video, tmpl='%06d.jpg'):
    os.system('ffmpeg -i ' + VIDEO_ROOT + '/' + video + ' -vf scale=256:256 ' + FRAME_ROOT + '/' + video[:-5] + '/' + tmpl) # python 3.5 compatible

def target(video_list):
    for video in video_list:
        event_dir = os.path.join(FRAME_ROOT, video[:-5])
        if not os.path.exists(event_dir):
            os.makedirs(event_dir)
        frame_list = os.listdir(event_dir)
        # pdb.set_trace()
        if len(frame_list) > 0:
            print(len(frame_list))
        else:
            extract(video)


if not os.path.exists(VIDEO_ROOT):
    raise ValueError('Please download videos and set VIDEO_ROOT variable.')
if not os.path.exists(FRAME_ROOT):
    os.makedirs(FRAME_ROOT)

video_list = sorted(os.listdir(VIDEO_ROOT), key=lambda x: int(x[:-5]))
# print(video_list)
splits = list(split(video_list, NUM_THREADS))

for i, split in enumerate(splits):
    target(split)

# threads = []
# for i, split in enumerate(splits):
#     thread = threading.Thread(target=target, args=(split,))
#     thread.start()
#     threads.append(thread)
#
# for thread in threads:
#     thread.join()