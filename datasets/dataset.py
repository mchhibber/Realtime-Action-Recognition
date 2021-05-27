import torchvision.datasets.hmdb51


def get_dataset_info(dataset_name):
    video_path = None
    annotation_path = None
    classes = None

    if dataset_name == "HMDB51":
        video_path = 'datasets/HMDB/video_data/'
        annotation_path = 'datasets/HMDB/test_train_splits'
        classes = 51
    else:
        video_path = 'datasets/UCF101/UCF-101/'
        annotation_path = 'datasets/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/'
        classes = 101
    return video_path, annotation_path, classes


def get_dataset(video_path, annotation_path, dataset_name, train_transforms, test_transforms, train):
    num_frames = 16  # 16
    clip_steps = 50
    num_workers = 2
    dataset = None
    if dataset_name == "HMDB51":
        if train:
            dataset = torchvision.datasets.HMDB51(video_path, annotation_path, num_frames,
                                                  step_between_clips=clip_steps, fold=1, train=True,
                                                  transform=train_transforms, num_workers=num_workers)
        else:
            dataset = torchvision.datasets.HMDB51(video_path, annotation_path, num_frames,
                                                  step_between_clips=clip_steps, fold=1, train=False,
                                                  transform=test_transforms, num_workers=num_workers)
    elif dataset_name == "UCF101":
        if train:
            dataset = torchvision.datasets.UCF101(video_path, annotation_path, num_frames,
                                                  step_between_clips=clip_steps, fold=1, train=True,
                                                  transform=train_transforms, num_workers=num_workers)
        else:
            dataset = torchvision.datasets.UCF101(video_path, annotation_path, num_frames,
                                                  step_between_clips=clip_steps, fold=1, train=False,
                                                  transform=test_transforms, num_workers=num_workers)
    return dataset
