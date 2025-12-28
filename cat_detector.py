"""
Cat Face Detector using OpenCV and Haar Cascades
Detects cat faces in images and videos
Author: SMJ
"""

import cv2
import argparse
import os
from datetime import datetime

def detect_cats_in_image(image_path, cascade_path, output_dir="outputs"):
    """
    Detect cat faces in a single image
    
    Args:
        image_path: Path to input image
        cascade_path: Path to Haar cascade XML file
        output_dir: Directory to save output image
    """
    # Load the cascade classifier
    cat_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect cat faces
    cats = cat_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=10,
        minSize=(75, 75)
    )
    
    # Draw rectangles around detected cats
    for (i, (x, y, w, h)) in enumerate(cats):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            image, 
            f"Cat #{i+1}", 
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    
    # Add detection count
    cv2.putText(
        image,
        f"Cats Detected: {len(cats)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, image)
    print(f"✓ Detected {len(cats)} cat(s) in image")
    print(f"✓ Saved output to: {output_path}")
    
    # Display the result
    cv2.imshow("Cat Detection - Press any key to close", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_cats_in_video(video_path, cascade_path, output_dir="outputs", save_video=True):
    """
    Detect cat faces in a video file or webcam
    
    Args:
        video_path: Path to video file or 0 for webcam
        cascade_path: Path to Haar cascade XML file
        output_dir: Directory to save output video
        save_video: Whether to save the output video
    """
    # Load the cascade classifier
    cat_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Open video capture
    if video_path == "0" or video_path == 0:
        cap = cv2.VideoCapture(0)
        print("Using webcam...")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"Processing video: {video_path}")
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if saving
    out = None
    if save_video and video_path != "0" and video_path != 0:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"detected_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")
    
    frame_count = 0
    total_detections = 0
    
    print("\n▶ Processing... Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect cat faces
        cats = cat_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=10,
            minSize=(75, 75)
        )
        
        if len(cats) > 0:
            total_detections += len(cats)
        
        # Draw rectangles around detected cats
        for (i, (x, y, w, h)) in enumerate(cats):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"Cat #{i+1}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        
        # Add frame info
        cv2.putText(
            frame,
            f"Cats: {len(cats)} | Frame: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Write frame to output video
        if out is not None:
            out.write(frame)
        
        # Display the frame
        cv2.imshow("Cat Detection - Press 'q' to quit", frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if out is not None:
        out.release()
        print(f"\n✓ Video saved successfully!")
    
    cv2.destroyAllWindows()
    
    print(f"\nStatistics:")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Total cat detections: {total_detections}")
    if frame_count > 0:
        print(f"   Average detections per frame: {total_detections/frame_count:.2f}")


def main():
    """Main function to parse arguments and run detection"""
    
    parser = argparse.ArgumentParser(
        description="Detect cat faces in images or videos using OpenCV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Detect cats in an image
            python cat_detector.py --input cat.jpg --mode image
  
            # Detect cats in a video
            python cat_detector.py --input cat_video.mp4 --mode video
  
            # Use webcam for live detection
            python cat_detector.py --input 0 --mode video
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input image/video, or 0 for webcam"
    )
    
    parser.add_argument(
        "-m", "--mode",
        choices=["image", "video"],
        default="video",
        help="Detection mode: 'image' or 'video' (default: video)"
    )
    
    parser.add_argument(
        "-c", "--cascade",
        default="haarcascade_frontalcatface.xml",
        help="Path to Haar cascade XML file (default: haarcascade_frontalcatface.xml)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output video (live preview only)"
    )
    
    args = parser.parse_args()
    
    # Check if cascade file exists
    if not os.path.exists(args.cascade):
        print(f"Error: Cascade file not found: {args.cascade}")
        print("\nPlease download haarcascade_frontalcatface.xml from:")
        print("   https://github.com/opencv/opencv/tree/master/data/haarcascades")
        return
    
    print("=" * 60)
    print("ฅ⁠^⁠•⁠ﻌ⁠•⁠^⁠ฅ CAT FACE DETECTOR")
    print("=" * 60)
    
    # Run detection based on mode
    if args.mode == "image":
        detect_cats_in_image(args.input, args.cascade, args.output)
    else:
        detect_cats_in_video(
            args.input,
            args.cascade,
            args.output,
            save_video=not args.no_save
        )
    
    print("\n✅ Detection complete!")


if __name__ == "__main__":
    main()
