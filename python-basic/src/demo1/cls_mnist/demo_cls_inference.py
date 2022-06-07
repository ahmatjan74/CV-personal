import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import cv2

# net

test_data = dataset.MNIST(root="mnist",
                          train=False,
                          transform=transforms.ToTensor(),
                          download=False)

test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=True)

cnn = torch.load("model/model_mnist.pkl")
# loss
# eval/test
loss_test = 0
accuracy = 0

for i, (images, labels) in enumerate(test_loader):
    outputs = cnn(images)
    _, pred = outputs.max(1)
    accuracy += (pred == labels).sum().item()

    images = images.numpy()
    labels = labels.numpy()
    pred = pred.numpy()
    # batchsize * 1 * 28 * 28

    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_label = labels[idx]
        im_pred = pred[idx]
        # 3通道数据，维度转换
        im_data = im_data.transpose(1, 2, 0)
        cv2.imshow('{}'.format(im_label), im_data)
        cv2.waitKey(0)
accuracy = accuracy / len(test_data)
print(accuracy)
