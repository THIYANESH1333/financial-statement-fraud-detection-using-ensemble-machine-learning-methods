# How to Open and Use This Project in Cisco Packet Tracer

## 📋 Prerequisites
- **Cisco Packet Tracer 7.3 or later** installed on your computer
- **Basic understanding** of networking concepts
- **Administrator privileges** (for installation if needed)

## 🚀 Step-by-Step Instructions

### Step 1: Install Cisco Packet Tracer (if not already installed)

#### Option A: Download from Cisco NetAcad
1. **Go to**: [Cisco NetAcad](https://www.netacad.com/)
2. **Sign up** for a free account
3. **Navigate to**: Resources → Packet Tracer
4. **Download** the latest version (7.3+ recommended)
5. **Install** following the installation wizard

#### Option B: Use Existing Installation
- **Check version**: Help → About Packet Tracer
- **Update if needed**: Download latest version from Cisco

### Step 2: Open Cisco Packet Tracer

1. **Launch Packet Tracer** from your desktop or start menu
2. **Wait for startup** (may take 30-60 seconds)
3. **You'll see the main interface** with device palette on the left

### Step 3: Create New Project

#### Method 1: Create from Scratch (Recommended)
1. **Click**: `File` → `New` (or Ctrl+N)
2. **Choose**: `Blank Project`
3. **Save as**: Navigate to your project folder
4. **Name it**: `Network_Topology.pkt`
5. **Click**: `Save`

#### Method 2: Use Template (Alternative)
1. **Click**: `File` → `New`
2. **Choose**: `From Template`
3. **Select**: `Basic Network` or `Empty`
4. **Click**: `OK`

### Step 4: Add Devices to Workspace

#### Add Main City Routers
1. **From Device Palette** (left side), select **Routers**
2. **Drag 3x Router 2811** to workspace:
   - **Delhi**: Position at top-center
   - **Mumbai**: Position at bottom-left  
   - **Chennai**: Position at bottom-right

#### Add Intermediate Devices
1. **Select Switches** from device palette
2. **Drag 6x Switch 2960** to workspace:
   - **HOP1**: Between Chennai and center
   - **HOP2**: Between center and Mumbai
   - **HOP3**: Between Delhi and center (right side)
   - **HOP4**: Continue the chain from HOP3
   - **HOP5**: Continue the chain from HOP4
   - **HOP6**: Between HOP5 and Chennai

### Step 5: Connect the Devices

#### Connect Main Cities (Direct Links)
1. **Click**: `Connections` tool (cable icon)
2. **Select**: `Copper Straight-Through` cable
3. **Connect**:
   - **Delhi Fa0/0** ↔ **Mumbai Fa0/0**
   - **Delhi Fa0/1** ↔ **Chennai Fa0/0**

#### Connect Multi-hop Paths
1. **Chennai to Mumbai path**:
   - **Chennai Fa0/1** ↔ **HOP1 Fa0/0**
   - **HOP1 Fa0/1** ↔ **HOP2 Fa0/0**
   - **HOP2 Fa0/1** ↔ **Mumbai Fa0/1**

2. **Delhi to Chennai path**:
   - **Delhi Fa1/0** ↔ **HOP3 Fa0/0**
   - **HOP3 Fa0/1** ↔ **HOP4 Fa0/0**
   - **HOP4 Fa0/1** ↔ **HOP5 Fa0/0**
   - **HOP5 Fa0/1** ↔ **HOP6 Fa0/0**
   - **HOP6 Fa0/1** ↔ **Chennai Fa1/0**

### Step 6: Apply Configurations

#### Configure Delhi Router
1. **Click** on Delhi router
2. **Go to**: `CLI` tab
3. **Copy** the entire content from `Configurations/Delhi_Router.txt`
4. **Paste** into CLI window
5. **Press Enter** after each command
6. **Wait** for configuration to complete

#### Configure Mumbai Router
1. **Click** on Mumbai router
2. **Go to**: `CLI` tab
3. **Copy** content from `Configurations/Mumbai_Router.txt`
4. **Paste** and press Enter
5. **Wait** for completion

#### Configure Chennai Router
1. **Click** on Chennai router
2. **Go to**: `CLI` tab
3. **Copy** content from `Configurations/Chennai_Router.txt`
4. **Paste** and press Enter
5. **Wait** for completion

#### Configure Intermediate Routers (HOP1-HOP6)
1. **Repeat the process** for each HOP router
2. **Use corresponding** configuration files
3. **Apply configurations** in order: HOP1 → HOP2 → HOP3 → HOP4 → HOP5 → HOP6

### Step 7: Test the Network

#### Basic Connectivity Test
1. **Click** on Delhi router
2. **Go to**: `CLI` tab
3. **Type**: `ping 192.168.1.2` (test Delhi to Mumbai)
4. **Press Enter**
5. **Should see**: `!!!!!` (successful ping)

#### Test All Connections
```bash
# From Delhi router
ping 192.168.1.2    # Delhi to Mumbai
ping 192.168.2.2    # Delhi to Chennai

# From Mumbai router  
ping 192.168.1.1    # Mumbai to Delhi
ping 192.168.2.2    # Mumbai to Chennai

# From Chennai router
ping 192.168.2.1    # Chennai to Delhi
ping 192.168.1.2    # Chennai to Mumbai
```

#### Check OSPF Status
```bash
# On any router
show ip ospf neighbor
show ip route
show ip ospf database
```

### Step 8: Save Your Project

1. **Click**: `File` → `Save` (or Ctrl+S)
2. **Choose location**: Your project folder
3. **Name**: `Network_Topology.pkt`
4. **Click**: `Save`

## 🔧 Troubleshooting Common Issues

### Issue 1: "Device not responding"
**Solution**:
- Check if device is powered on (green light)
- Verify cable connections
- Restart the device

### Issue 2: "Configuration not applying"
**Solution**:
- Ensure you're in CLI mode
- Copy-paste entire configuration
- Press Enter after each command
- Wait for completion

### Issue 3: "Ping failures"
**Solution**:
- Check IP addressing
- Verify routing table: `show ip route`
- Check OSPF neighbors: `show ip ospf neighbor`

### Issue 4: "OSPF neighbors not forming"
**Solution**:
- Check interface configurations
- Verify IP addresses and subnet masks
- Ensure interfaces are up: `no shutdown`

## 📊 Verification Commands

### Check Network Status
```bash
# On any router
show ip interface brief    # Check interface status
show ip route              # Check routing table
show ip ospf neighbor      # Check OSPF neighbors
show ip ospf database      # Check OSPF database
```

### Test Connectivity
```bash
ping [destination_ip]      # Test connectivity
traceroute [destination]   # Trace network path
```

### Debug Commands
```bash
debug ip ospf events       # Debug OSPF
debug ip ospf adj          # Debug OSPF adjacency
```

## 🎯 Next Steps

### 1. Run Automated Tests
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run connectivity tests
python Scripts/test_connectivity.py

# Run network analysis
python Scripts/network_analysis.py
```

### 2. Explore Advanced Features
- **Simulation Mode**: Test packet flow
- **Performance Monitoring**: Check bandwidth usage
- **Load Balancing**: Test multiple paths
- **Failover Testing**: Simulate link failures

### 3. Customize the Network
- **Modify routing costs**
- **Add more devices**
- **Implement security features**
- **Test different scenarios**

## 📚 Additional Resources

### Documentation
- **Setup_Guide.md**: Detailed setup instructions
- **Configuration_Guide.md**: Technical documentation
- **Testing_Procedures.md**: Complete testing methodology

### Support
- **Cisco Documentation**: Official Packet Tracer guides
- **Community Forums**: Help and troubleshooting
- **Video Tutorials**: YouTube and Cisco Learning Network

## ✅ Success Checklist

- [ ] Packet Tracer installed and running
- [ ] All devices added to workspace
- [ ] All connections made properly
- [ ] All configurations applied
- [ ] Basic connectivity working
- [ ] OSPF neighbors formed
- [ ] Project saved successfully

---

**🎉 Congratulations! You now have a fully functional network topology in Cisco Packet Tracer!**

**Next**: Follow the `Quick_Start_Guide.md` for advanced features and testing procedures.
