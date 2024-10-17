import os
import cv2
import torch

class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        
    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

def plot_box_labl(preds: list, ori_ims: list, line_width=None, save_dir=None):
    colors = Colors() 
    lw = line_width or max(round(sum(ori_ims[0].shape) / 2 * 0.003), 2)
    for i, pred in enumerate(preds):
        boxes = pred[:, :4].to(torch.int).tolist()
        conf = pred[:, -2].tolist()
        labels = pred[:, -1].to(torch.int).tolist()
        for j in range(len(boxes)):
            # box plot
            p_lt, p_rb = (boxes[j][0], boxes[j][1]), (boxes[j][2], boxes[j][3])
            color = colors(labels[j])
            cv2.rectangle(ori_ims[i], p_lt, p_rb, color, thickness=lw, lineType=cv2.LINE_AA)
        
            # label show
            lb = str(labels[j]) + str(round(conf[j], 2))
            txt_color = (255, 255, 255)
            w, h = cv2.getTextSize(lb, 0, fontScale=lw / 3, thickness=1)[0]  # text width, height
            outside = p_lt[1] - h >= 3
            p_rb = p_lt[0] + w, p_lt[1] - h - 3 if outside else p_lt[1] + h + 3
            cv2.rectangle(ori_ims[i], p_lt, p_rb, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(ori_ims[i],
                        lb,
                        (p_lt[0], p_lt[1] - 2 if outside else p_lt[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                        )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i, im in enumerate(ori_ims):
            cv2.imwrite(os.path.join(save_dir, 'img%d.jpg' % i), im) 
