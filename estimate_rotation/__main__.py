from estimate_rotation import dataset

def main():
    # user params
    train = '/home/bogdan/work/visionsemantics/data/imagenet'
    # ~user_params

    dataset.load(train)


if __name__ == '__main__':
    main()