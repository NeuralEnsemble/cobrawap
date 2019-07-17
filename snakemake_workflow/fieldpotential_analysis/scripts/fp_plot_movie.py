import os
import sys
import argparse
import matplotlib.pyplot as plt
import imageio
import subprocess
import neo
sys.path.append(os.getcwd())


def movie_maker(frames_directory, movie_directory, frame_format, frame_name,
                movie_name, vid_format,
                frames_per_sec=20, quality=23, scale_x=1920, scale_y=1080):
    """
    Makes a movie from a given frame series.

    Parameters
    ==========
    frames_directory : str
        Directory where all frames of the movie are located
    movie_directory : str
        Directory in which movie should be saved
    frame_format : str
        Frame format as string (e.g. 'png')
    movie_name : str
        Filename of the movie (without extension)
    frames_per_sec : int
        Number of frames per second. Default: 20
    quality
        This is the crf argument of avconv. Here, higher is smaller file size
        (0=lossless, 23=average, 51=worst quality). Default: 23
    scale_x, scale_y : int
        X and Y resolution of movie. Default: Full HD (1920x1080)
    """

    command = (
        'ffmpeg',
        '-y',  # Overwrite existing file
        # '-v', 'quiet',
        '-i', frames_directory + os.path.sep + \
        frame_name + '_%05d.' + frame_format,
        # '-c', 'h264',  # MP4 Codec
        '-q', str(quality),
        '-crf', str(quality),  # Quality
        '-vf', 'scale=' + str(scale_x) + ':' + str(scale_y),  # Resolution
        '-r', str(frames_per_sec),
        movie_directory + os.path.sep + movie_name + '.' + vid_format)

    print("\n\nExecuting:\n %s\n\n" % ' '.join(command))
    return command


if __name__ == '__main__':

    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--logMUA",        nargs='?', type=str)
    CLI.add_argument("--out_movie",     nargs='?', type=str)
    CLI.add_argument("--frame_folder",  nargs='?', type=str)
    CLI.add_argument("--frame_name",    nargs='?', type=str)
    CLI.add_argument("--fps",           nargs='?', type=int, default=10)
    CLI.add_argument("--frame_format",  nargs='?', type=str)
    CLI.add_argument("--scale_x",       nargs='?', type=int)
    CLI.add_argument("--scale_y",       nargs='?', type=int)
    CLI.add_argument("--quality",       nargs='?', type=int)
    CLI.add_argument("--vid_format",    nargs='?', type=str, default='mp4')

    args = CLI.parse_args()

    with neo.NixIO(args.image_file) as io:
        logMUA = io.read_block().segments[0].analogsignals

    shape = [len(logMUA[0].times,
             logMUA[0].annotations[grid_size][0],
             logMUA[0].annotations[grid_size][1]]
    logMUA_array = np.zeros(shape)
    for asig in logMUA:
        x, y = asig.annotations['coordiantes']
        logMUA_array[:,x,y] = asig.as_array()

    if not os.path.exists(args.frame_folder):
        os.makedirs(args.frame_folder)

    for num, img in enumerate(logMUA_array):
        fig, ax = plt.subplots()
        ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('{} {}'.format(logMUA[0].times[num],
                                    logMUA[0].times.units.dimensionality.string))
        plt.savefig(os.path.join(args.frame_folder,
                                 args.frame_name
                                 + '_{}.{}'.format(str(num).zfill(5),
                                                   args.frame_format)))
        plt.close(fig)

    movie_dir, movie_name = os.path.split(args.out_movie)

    subprocess.call(movie_maker(frames_directory=args.frame_folder,
                                frame_name=args.frame_name,
                                movie_directory=movie_dir,
                                movie_name=movie_name.split('.')[0],
                                frame_format=args.frame_format,
                                frames_per_sec=args.fps,
                                quality=args.quality,
                                scale_x=args.scale_x,
                                scale_y=args.scale_y,
                                vid_format=args.vid_format)
                    )
