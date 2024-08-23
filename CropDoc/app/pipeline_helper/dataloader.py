import torch


class TransformDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, transform, *args, **kwargs):
        self.dataset = dataset
        self.transform = transform
        super().__init__(dataset, *args, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        transformed_batch = []
        for item in batch:
            img, crop_label, state_label = item
            if self.transform:
                img = self.transform(img)
            transformed_batch.append((img, crop_label, state_label))
        return torch.utils.data.dataloader.default_collate(transformed_batch)