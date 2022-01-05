# Visual-Information-Processing

Eczema is a chronic skin disease that when left untreated
may lead to major health consequences if not detected and
controlled early. Early detection and seeking out medical
attention early may help prevent the skin from worsening.
However, diagnosis of eczema is time consuming and
costly. This paper aims to detect and classify eczema on the
skin using modern image processing techniques and
machine learning algorithms. In this paper, two methods of
comparison are used to detect and classify the skin disease
namely non-segmented and segmented methods. The
non-segmented method achieves an accuracy score 59%
while segmented method achieves an accuracy score of
52%. The results on which method is more suitable are
skewed towards non-segmented methods.

<hr>

In this paper, we did the Eczema classification
using the Support Vector Machine method. Before the actual
classification, multiple stages of image pre-processing and
feature extraction were conducted on the raw image. The
image pre-processing includes occlusion removal, LUV
color space, morphology, thresholding, median filters, and
Contrast-Limited Adaptive Histogram Equalization. After
image pre-processing, image segmentation was performed
to segment between the skin and background before moving
on to the features extraction. The feature extraction extracts
the color feature to use for classification. Finally, the
classification was conducted on two different datas, one for
non-segmented data while the other segmented data. This is
to see which situation does the classification perform better.
The classification classifies the image into acute and
chronic.

<hr>

After performing all kind of image cleaning process, the final output looked like below


![alt text](https://github.com/Arpi33/Visual-Information-Processing/blob/main/img/final%20output.PNG?raw=true)

