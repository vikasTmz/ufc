'''

Script to prepare dataset

'''

import os, os.path
import numpy as np
import cv2
import argparse
import random

VALID_EXT = [".jpg", ".png"]



class Dataset():
    def __init__(self):
        self.train_size = 900
        self.val_size = 100
        self.test_size = 100
        self.train = 0
        self.val = 0
        self.test = 0
        self.imgs = []
        self.Y = 0
        self.X = 0
        self.H = 256
        self.W = 256
        self.isOccluded = []

    def preprocess_image(self, img, category, img_id):
        # crop_img = img[self.Y:self.Y + self.H, self.X:self.X + self.W]
        resize_img = cv2.resize(img, (self.W, self.H), interpolation = cv2.INTER_AREA)
        if category == 'A':
            if A.isOccluded[img_id] == 1:
                rec_h = random.randint(10, self.H / 3)
                rec_w = random.randint(10, self.W / 3)
                x1 = random.randint(0, self.W - rec_w - 1)
                y1 = random.randint(0, self.H - rec_h - 1)
                random_img = cv2.imread(get_random_image())
                resize_img[y1:y1 + rec_h, x1: x1 + rec_w] = cv2.resize(random_img, (rec_w, rec_h), interpolation = cv2.INTER_AREA)

        return resize_img

    def create_train(self, folder, image, category):
        self.train += 1
        if self.train > self.train_size:
            self.create_val(folder, image, category)
        else:
            img = cv2.imread(os.path.join(image))
            final_img = self.preprocess_image(img, category, self.train)
            cv2.imwrite(os.path.join(folder, 'train', str(self.train) + '.png'), final_img)

    def create_val(self, folder, image, category):
        self.val += 1
        if self.val > self.val_size:
            self.create_test(folder, image, category)
        else:
            img = cv2.imread(os.path.join(image))
            final_img = self.preprocess_image(img, category, self.val)
            cv2.imwrite(os.path.join(folder, 'val', str(self.val) + '.png'), final_img)

    def create_test(self, folder, image, category):
        self.test += 1
        img = cv2.imread(os.path.join(image))
        final_img = self.preprocess_image(img, category, self.test)
        cv2.imwrite(os.path.join(folder, 'test', str(self.test) + '.png'), final_img)

def get_random_image():
    root = 'dtd/images/'
    sub_dir = os.listdir(root)
    random.shuffle(sub_dir)
    images = os.listdir(os.path.join(root, sub_dir[0]))
    random.shuffle(images)
    return os.path.join(root, sub_dir[0], images[0])

def get_images_in_directory(path, domainA, domainB):
    domainA_imgs = []
    domainB_imgs = []
    for d in os.listdir(path):
        for f in os.listdir(os.path.join(path, d)):
            f_split = f.split("_")[-1].split('.')[0]
            if domainB == f_split:
                domainB_imgs.append(os.path.join(path, d, f))

            if domainA == "rgb" and f_split == "0":
                domainA_imgs.append(os.path.join(path, d, f))

            # ext = f_split[1]
            # if ext.lower() not in VALID_EXT:
            #     continue

    return [domainA_imgs, domainB_imgs]

if __name__ == "__main__":

    # Parse CLI arguments
    parser = argparse.ArgumentParser('create image pairs')
    parser.add_argument('--root', dest='root', help='root directory containing A and B folder created by blender script', type=str, default='./')
    parser.add_argument('--output_dir', dest='output_dir', help='output dir for pix2pix', type=str, default='ir2depth_dataset')
    parser.add_argument('--train_size', dest='train_size', help='size of training set', type=int, default=1000)
    parser.add_argument('--val_size', dest='val_size', help='size of validation set', type=int, default=100)
    parser.add_argument('--test_size', dest='test_size', help='size of testing set', type=int, default=100)
    parser.add_argument('--X', dest='X', help='crop X value', type=int, default=0)
    parser.add_argument('--Y', dest='Y', help='crop Y value', type=int, default=0)
    parser.add_argument('--W', dest='W', help='crop Width', type=int, default=256)
    parser.add_argument('--domainA', dest='domainA', help='Domain A type', type=str, default='rgb')
    parser.add_argument('--domainB', dest='domainB', help='Domain B type', type=str, default='depth')
    parser.add_argument('--occlusion', dest='occlusion', help='add occlusions', type=str, default=False)

    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] = ' % arg,  getattr(args, arg))


    os.system('mkdir ' + args.root + '/A ' + args.root + '/B')

    fold_A = os.path.join(args.root, 'A')
    os.system('mkdir ' + os.path.join(fold_A, 'train') + ' ' + os.path.join(fold_A, 'test') + ' ' + os.path.join(fold_A, 'val'))

    fold_B = os.path.join(args.root, 'B')
    os.system('mkdir ' + os.path.join(fold_B, 'train') + ' ' + os.path.join(fold_B, 'test') + ' ' + os.path.join(fold_B, 'val'))

    fold_AB = os.path.join(args.root, args.output_dir)
    use_AB = False

    A = Dataset()
    B = Dataset()

    [A.imgs, B.imgs] = get_images_in_directory(args.root, args.domainA, args.domainB)
    # B.imgs = get_images_in_directory(fold_B)

    A.train_size = B.train_size = args.train_size
    A.val_size = B.val_size = args.val_size
    A.test_size = B.test_size = args.test_size
    A.Y = B.Y = args.Y
    A.X = B.X = args.X
    A.W = B.W = A.H = B.H = args.W

    if args.occlusion:
        A.isOccluded = [1] * (A.train_size / 3) + [0] * (A.train_size - (A.train_size / 3) + 1)
        random.shuffle(A.isOccluded)


    # Create Domain A (default RGB) set containing train, val and test dir.
    for img in A.imgs:
        A.create_train(fold_A, img, 'A')

    # Create Domain B (default Depth) set containing train, val and test dir.
    for img in B.imgs:
        B.create_train(fold_B, img, 'B')


    # Combine domain A and B into one for pix2pix.

    splits = ['test','train','val']

    for sp in splits:
        img_fold_A = os.path.join(fold_A, sp)
        img_fold_B = os.path.join(fold_B, sp)
        img_list = os.listdir(img_fold_A)
        if use_AB:
            img_list = [img_path for img_path in img_list if '_A.' in img_path]

        num_imgs = len(img_list)
        print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
        img_fold_AB = os.path.join(fold_AB, sp)
        if not os.path.isdir(img_fold_AB):
            os.makedirs(img_fold_AB)
        print('split = %s, number of images = %d' % (sp, num_imgs))
        for n in range(num_imgs):
            name_A = img_list[n]
            path_A = os.path.join(img_fold_A, name_A)
            if use_AB:
                name_B = name_A.replace('_A.', '_B.')
            else:
                name_B = name_A
            path_B = os.path.join(img_fold_B, name_B)
            if os.path.isfile(path_A) and os.path.isfile(path_B):
                name_AB = name_A
                if use_AB:
                    name_AB = name_AB.replace('_A.', '.') # remove _A
                path_AB = os.path.join(img_fold_AB, name_AB)
                im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
                im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)
