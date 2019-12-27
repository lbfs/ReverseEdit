# ReverseEdit
A video remastering tool for recreating complex edits with higher quality source footage.

![Sample](https://i.imgur.com/Ino2yVE.png)

A sample recreation is displayed above demonstrating the lower quality edit of the source footage on the left, being recreated out of the source video. The image is shown is from a short film called Ark. The top left value shown on the edited frame is the structural similiarity index, a best-effort metric that will be used for comparing the overall accuracy of the video recreation. 

## How does it work?

The project is using a combination of image hashing algorithms, combined with a fast-lookup using a specialized metric tree called a vantage point tree using the hamming distance as a distance metric. The tool finds matches between all frames, uses the average distance between frames to create splits, and then exports those splits to a video editor like OpenShot. 
