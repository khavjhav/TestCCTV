import cv2

def get_gpu_info():
    try:
        num_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if num_devices == 0:
            print("No GPU devices found.")
            return

        print(f"Number of GPU devices: {num_devices}")

        for device_id in range(num_devices):
            device_info = cv2.cuda.Device(device_id).getDeviceInfo()
            print(f"Device {device_id} - {device_info['name']}")
            print(f"  Total Memory: {device_info['totalMemory']} bytes")
            print(f"  MultiProcessor Count: {device_info['multiProcessorCount']}")
            print(f"  Compute Capability: {device_info['majorVersion']}.{device_info['minorVersion']}")
            print("")

    except Exception as e:
        print(f"Error: {e}")

get_gpu_info()