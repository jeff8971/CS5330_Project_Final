import cv2
print(cv2.__version__)
# Check if the specific tracker can be created directly
if hasattr(cv2, 'TrackerKCF_create'):
    tracker = cv2.TrackerKCF_create()
else:
    # If the direct create function is not available, access it through cv2.TrackerKCF
    tracker = cv2.TrackerKCF.create()

  # This will help verify which version of OpenCV is active

# Attempt to create a KCF Tracker
try:
    tracker = cv2.TrackerKCF_create()
    print("Tracker created successfully!")
except AttributeError as e:
    print(f"Failed to create tracker: {str(e)}")
