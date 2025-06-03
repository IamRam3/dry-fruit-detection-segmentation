class COCOObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']

        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes, labels = [], []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)
    

class dataset_dir:
  def __init__(self,path):
    self.path = path
    self.train_dir = self.path + "/train"
    self.val_dir = self.path + "/valid"
    self.test_dir = self.path + "/test"
    self.annFile = "/_annotations.coco.json"
    self.train_annFile = self.train_dir + self.annFile
    self.val_annFile = self.val_dir + self.annFile
    self.test_annFile = self.test_dir + self.annFile
  
  def create_custom_datasets(self, get_transform):
    train_dataset = COCOObjectDetectionDataset(
        root=self.train_dir,
        annFile=self.train_annFile,
        transforms=get_transform()
        )
    val_dataset = COCOObjectDetectionDataset(
        root=self.val_dir,
        annFile=self.val_annFile,
        transforms=get_transform()
    ) 
    test_dataset = COCOObjectDetectionDataset(
        root=self.test_dir,
        annFile=self.test_annFile,
        transforms=get_transform()
    )
    return train_dataset, val_dataset, test_dataset


class CustomDataLoader:
  def __init__(self, train_dataset, val_dataset, test_dataset, batch_size):
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.test_dataset = test_dataset
    self.batch_size = batch_size

  def create_data_loader(self):
    train_data_loader = torch.utils.data.DataLoader(
        self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    val_data_loader = torch.utils.data.DataLoader(
        self.val_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    test_data_loader = torch.utils.data.DataLoader(
        self.test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    return train_data_loader, val_data_loader, test_data_loader