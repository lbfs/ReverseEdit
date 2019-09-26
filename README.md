# ReverseEdit
A video remastering tool for recreating complex edits with higher quality source footage.

**NOTE**: This project is currently a work-in-progress, do not expect anything to work. I'm building this for my capstone project as part of my Computer Science degree.

![Sample](https://i.imgur.com/Ino2yVE.png)

A sample recreation is displayed above demonstrating the lower quality edit of the source footage on the left, being recreated out of the source video. The image is shown is from a short film called Ark. The top left value shown on the edited frame is the structural similiarity index, a best-effort metric that will be used for comparing the overall accuracy of the video recreation. 

## How does it work?

The project is using a combination of image hashing algorithms, combined with a fast-lookup using a specialized metric tree called a vantage point tree using the hamming distance as a distance metric. 
For each clip that will be processed, a tree will be built allowing for quick searching for the best possible match to the frames in that video. Then the best matches will all be coalessed into another vantage point tree for final lookup.
A small correction phase is performed with the structural similiarity index using a user-defined range that seeks around each matched frame in the source video, attempting to find a more structurally-sound frame. This is required as image hashing can only be used as a filter, as image hashing does not look for images that are conceptually similar.  

## Okay, you've scanned the video. Now what?

While not implemented yet, the idea is to allow export to a video editor for further editing by the person needing to remaster the content, or directly to a video file. 

## When do you expect this project to be in a usable state? 

Unknown, but I am aiming for the base functionality to be implemented within the next two months, with further usability and performance improvements to be added on after. 
