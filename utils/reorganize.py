from mmcv.parallel import DataContainer as DC
import torch

def reorganize_data(data_batch, num_images_3dce, num_slices):
    # img_metas_reorganize = []
    # gt_boxes_reorganize = []
    # gt_labels_reorganize = []
    if num_images_3dce == 1:
        return data_batch
    
    img_reorganize = []
    if isinstance(data_batch['img'], DC):

        for i in range(len(data_batch['img'].data)):

            img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore = data_batch['img'].data[i], data_batch['img_meta'].data[i], \
                                              data_batch['gt_bboxes'].data[i], data_batch['gt_labels'].data[i], data_batch['gt_bboxes_ignore']
            samples_per_gpu = img.size(0)
            num_images_3dce = num_images_3dce
            num_slices = num_slices  # 3

            img_new = torch.empty((samples_per_gpu * num_images_3dce, num_slices, img.size(2), img.size(3)))
            # img_metas_new = []
            # gt_bboxes_new = []
            # gt_labels_new = []
            for p in range(samples_per_gpu):
                for q in range(num_images_3dce):
                    img_new[p * num_images_3dce + q, :, :, :] = img[p, q * num_slices:(q + 1) * num_slices, :, :]
                    # img_metas_new.append(img_metas[p])
                    # if q == (num_images_3dce - 1) / 2:
                    #     gt_bboxes_new.append(gt_bboxes[p])
                    #     gt_labels_new.append(gt_labels[p])
                    # else:
                    #     gt_bboxes_new.append(torch.Tensor())
                    #     gt_labels_new.append(torch.Tensor())
            img_reorganize.append(img_new)
            # img_metas_reorganize.append(img_metas_new)
            # gt_boxes_reorganize.append(gt_bboxes_new)
            # gt_labels_reorganize.append(gt_labels_new)
        # data_batch = dict(
        #                 img=DC(img_reorganize, stack=True),
        #                 img_meta=DC(img_metas_reorganize, cpu_only=True),
        #                 gt_bboxes=DC(gt_boxes_reorganize),
        #                 gt_labels=DC(gt_labels_reorganize)
        #             )
        data_batch = dict(
            img=DC(img_reorganize, stack=True),
            img_meta=data_batch['img_meta'],
            gt_bboxes=data_batch['gt_bboxes'],
            gt_labels=data_batch['gt_labels'],
            gt_bboxes_ignore=data_batch['gt_bboxes_ignore']
        )
    else:
        #for test
        for i in range(len(data_batch['img'])):
            img, img_metas = data_batch['img'][i], data_batch['img_meta'][i]
            samples_per_gpu = 1 
            num_images_3dce = num_images_3dce
            num_slices = num_slices  # 3
            if img.dim()==4:
                img_new = torch.empty((samples_per_gpu * num_images_3dce, num_slices, img.size(2), img.size(3)))
                for p in range(samples_per_gpu):
                    for q in range(num_images_3dce):
                        img_new[p * num_images_3dce + q, :, :, :] = img[p, q * num_slices:(q + 1) * num_slices, :, :]
            else:
                img_new = torch.empty((samples_per_gpu * num_images_3dce, num_slices, img.size(1), img.size(2)))
                for q in range(num_images_3dce):
                    img_new[q, :, :, :] = img[q * num_slices:(q + 1) * num_slices, :, :]
            img_reorganize.append(img_new)

        data_batch = dict(
            img=img_reorganize,
            img_meta=data_batch['img_meta'],
        )
    return data_batch
