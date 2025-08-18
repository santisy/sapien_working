# Articulated Object Annotation Tool

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get SAPIEN Token
- Register at https://sapien.ucsd.edu/downloads
- Click "Generate Python Access Token"
- Set environment variable:
```bash
export TOKEN_SAPIEN_STR="your_actual_token_here"
```

### 3. Start Server

**Local Use:**
```bash
python anno_tool.py
```
Open http://localhost:5000

**Server Use:**
```bash
# On server
python anno_tool.py

# Access from your local machine
# Replace SERVER_IP with actual server IP
http://SERVER_IP:5000
```

**Production (with Gunicorn):**
```bash
./start.sh prod
```

## How to Annotate

### Step 1: Load Object
1. Enter Object UID in left panel
2. Click "📋 Load Object Info"
3. Check joint information appears in right corner
4. ⚠️ Warning shows if UID already annotated

### Step 2: Test Video
1. Select a joint from dropdown
2. Adjust camera parameters (azimuth, elevation, distance)
3. Click "🎬 Generate Test Video"
4. Verify videos show correct movement

### Step 3: Save Annotation
1. Fill joint indices: `0,1`
2. Fill joint prompts: `"upper drawer","lower drawer"`
3. Set angle ranges (elevation/azimuth)
4. Write base prompt describing the movement
5. Click "💾 Save Annotation"

### Step 4: Export
Click "📥 Export as CSV" to download annotations for pandas.

## Network Access

- **Local**: http://localhost:5000
- **LAN**: http://YOUR_IP:5000 (allow port 5000 in firewall)
- **Server**: Change host in anno_tool.py to `0.0.0.0` (default)

## Files Structure
```
camera.py          # Video generation script
anno_tool.py       # Web server
anno.html         # Frontend (auto-loaded)
requirements.txt  # Dependencies
start.sh         # Production startup script
```