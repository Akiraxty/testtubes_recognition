import tensorflow as tf
import os
import numpy as np
import cv2
import glob


#参数设置
learning_rate = 0.0001
train_size = 350
batch_size = 50
train_data_path = "train_data"
test_data_path = "test_data"
img_num = 2
n_classes = 2
batch_step = 0
test1 = "test_data\\test1.jpg"


#读取图片，利用canny算子进行边缘检测
img = cv2.imread('tupian/171921-005.bmp')
img_yuan = cv2.resize(img, (1024, 544))
img = cv2.cvtColor(img_yuan, cv2.COLOR_BGR2GRAY)
img = cv2.Canny(img, 50, 130)


#利用霍夫变换检测圆
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=40, param1=220, param2=30, minRadius=5, maxRadius=40)


#在保存切割的图像之前，清空文件夹内的内容
fileNames = glob.glob(test_data_path + r'\*')
for fileName in fileNames:
    if fileName == test1:
        pass
    else:
        os.remove(fileName)


#对图像进行标注，分割，保存
for circle in circles[0]:
    x = int(circle[0])
    y = int(circle[1])
    r = int(circle[2])
    img_rect = cv2.rectangle(img, (x-r, y-r), (x+r, y+r), (0, 0, 255))
    img_rect_yuan = cv2.rectangle(img_yuan, (x-r, y-r), (x+r, y+r), (0, 0, 255))
    img_save = img_rect[y-r:y+r, x-r:x+r]
    img_save = cv2.resize(img_save, (64, 64))
    cv2.imwrite("test_data/test" + str(img_num) + ".jpg", img_save)
    cv2.imshow("ssi",img_rect_yuan)
    cv2.imshow("sss",img_rect)
    img_num += 1
print("请按任意键继续")
cv2.waitKey()



#提取测试图像
def get_test_images(test_data_path):
    test_images = cv2.imread(test_data_path+"/test1.jpg")
    test_images = test_images.reshape([1, 64*64*3])
    for files in os.listdir(test_data_path):
        files = cv2.imread(test_data_path+"/"+files)
        files = files.reshape([1, 64*64*3])
        test_images = np.vstack((test_images, files))
    return test_images


#读取训练图像
def get_file(train_data_path,batch_step):

    tubes = cv2.imread(train_data_path+"/1train76.jpg")
    tubes = tubes.reshape([1, 64*64*3])
    not_tubes = cv2.imread(train_data_path+"/0train209.jpg")
    not_tubes = not_tubes.reshape([1, 64*64*3])
    label_tubes = np.array([1])
    label_not_tubes = np.array([0])
    for file in os.listdir(train_data_path):
        name = file.split("t")
        if name[0] == "1":
            file1 = cv2.imread(train_data_path+"/"+file)
            file1 = file1.reshape([1, 64*64*3])
            tubes = np.vstack((tubes, file1))
            label_tubes = np.vstack((label_tubes, np.array([1])))
        else:
            file0 = cv2.imread(train_data_path+"/"+file)
            file0 = file0.reshape([1, 64*64*3])
            not_tubes = np.vstack((not_tubes, file0))
            label_not_tubes = np.vstack((label_not_tubes, np.array([0])))
    image_list = np.vstack((tubes, not_tubes))
    label_list = np.vstack((label_tubes, label_not_tubes))
    temp = np.hstack((image_list, label_list))
    np.random.shuffle(temp)
    images = temp[batch_step: batch_step + 50, 0:12288]
    labels = temp[batch_step: batch_step + 50, 12288:]
    labels = tf.one_hot(labels, 2, 1, axis=1)
    labels = tf.reshape(labels, [-1, 2])
    batch_step += 1
    return images, labels


#设置占位符
xs = tf.placeholder(tf.float32, [None, 64*64*3])
ys = tf.placeholder(tf.float32, [None, 2])
image = tf.reshape(xs, [-1, 64, 64, 3])


##CNN，训练是否为假圆
conv1 = tf.layers.conv2d(inputs=image, filters=16, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)     # ->64,64,16
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)    # ->32,32,16
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, "same", activation=tf.nn.relu)     # ->32,32,32
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # ->16,16,32
flat = tf.reshape(pool2, [-1, 16*16*32])
hidden1 = tf.layers.dense(flat, 64)
output = tf.layers.dense(hidden1, 2)

loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=output)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(ys, 1), predictions=tf.argmax(output, 1))[1]

sess = tf.Session()
init = tf.global_variables_initializer()
init_lo = tf.local_variables_initializer()
sess.run(init)
sess.run(init_lo)


#训练阶段
for step in range(train_size):
    sum = -1
    train_image, train_label = get_file(train_data_path,batch_step)
    train_label = sess.run(train_label)
    sess.run(train_op, {xs: train_image, ys: train_label})
    loss_ = sess.run(loss, {xs: train_image, ys: train_label})
    accuracy_ = sess.run(accuracy, {xs: train_image, ys: train_label})
    print("step:%d" % step, "| loss:%.4f" % loss_, "| accuracy:%.4f" % accuracy_)

    if step % 50 == 0:
        test_image = get_test_images(test_data_path)
        output_ = sess.run(output, {xs:test_image})
        for i in range(len(output_[:, 1])):
            if output_[i, 1] > output_[i, 0]:
                sum += 1
        print("试管数量为：%d" % sum)
