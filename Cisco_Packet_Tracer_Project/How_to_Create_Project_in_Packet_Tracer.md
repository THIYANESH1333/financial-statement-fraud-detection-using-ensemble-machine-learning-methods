# How to Create the Network Project in Cisco Packet Tracer

## ❌ The Issue
The `Network_Topology.pkt file I provided earlier is just a text placeholder, not a real Cisco Packet Tracer project file. You need to create the project manually in Packet Tracer.

## ✅ Solution: Create the Project Step-by-Step

### Step 1: Open Cisco Packet Tracer
1. **Launch Cisco Packet Tracer**
2. **Click**: `File` → `New` → `Blank Project`
3. **Save as**: `Network_Topology.pkt`

### Step 2: Add Devices to Workspace

#### Add Main City Routers (3x)
1. **From Device Palette** (left side), select **Routers**
2. **Drag 3x Router 2811** to workspace:
   - **Delhi**: Position at top-center
   - **Mumbai**: Position at bottom-left  
   - **Chennai**: Position at bottom-right

#### Add Intermediate Devices (6x)
1. **Select Switches** from device palette
2. **Drag 6x Switch 2960** to workspace:
   - **HOP1**: Between Chennai and center
   - **HOP2**: Between center and Mumbai
   - **HOP3**: Between Delhi and center (right side)
   - **HOP4**: Continue the chain from HOP3
   - **HOP5**: Continue the chain from HOP4
   - **HOP6**: Between HOP5 and Chennai

### Step 3: Connect the Devices

#### Connect Main Cities (Direct Links)
1. **Click**: `Connections` tool (cable icon)
2. **Select**: `Copper Straight-Through` cable
3. **Connect**:
   - **Delhi Fa0/0** ↔ **Mumbai Fa0/0** (Cost: 100)
   - **Delhi Fa0/1** ↔ **Chennai Fa0/0** (Cost: 100)

#### Connect Multi-hop Path 1 (Chennai → Mumbai)
1. **Chennai Fa0/1** ↔ **HOP1 Fa0/0** (Cost: 10, BW: 200)
2. **HOP1 Fa0/1** ↔ **HOP2 Fa0/0** (Cost: 10, BW: 20)
3. **HOP2 Fa0/1** ↔ **Mumbai Fa0/1** (Cost: 10)

#### Connect Multi-hop Path 2 (Delhi → Chennai)
1. **Delhi Fa1/0** ↔ **HOP3 Fa0/0** (Cost: 20)
2. **HOP3 Fa0/1** ↔ **HOP4 Fa0/0** (Cost: 20)
3. **HOP4 Fa0/1** ↔ **HOP5 Fa0/0** (Cost: 20)
4. **HOP5 Fa0/1** ↔ **HOP6 Fa0/0** (Cost: 10)
5. **HOP6 Fa0/1** ↔ **Chennai Fa1/0** (Cost: 10, BW: 170)

### Step 4: Apply Configurations

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

### Step 5: Test the Network

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

### Step 6: Save Your Project

1. **Click**: `File` → `Save` (or Ctrl+S)
2. **Choose location**: Your project folder
3. **Name**: `Network_Topology.pkt`
4. **Click**: `Save`

## 🔧 Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Device not responding | Check if device is powered on (green light) |
| Configuration not applying | Copy-paste entire config, press Enter |
| Ping failures | Check IP addressing, routing table |
| OSPF neighbors not forming | Check interface configs, IP addresses |

### Verification Commands
```bash
show ip interface brief    # Check interface status
show ip route              # Check routing table
show ip ospf neighbor      # Check OSPF neighbors
show ip ospf database      # Check OSPF database
```

## 📋 Quick Checklist

- [ ] Packet Tracer opened
- [ ] New project created
- [ ] All 9 devices added
- [ ] All connections made
- [ ] All configurations applied
- [ ] Basic connectivity working
- [ ] Project saved

## 🎯 Next Steps

1. **Test connectivity** between all cities
2. **Check OSPF neighbors** are formed
3. **Verify routing table** has all routes
4. **Run simulation mode** to see packet flow
5. **Test failover** by disconnecting links

---

**🎉 You now have a fully functional network topology in Cisco Packet Tracer!**

The project is ready to use for learning, testing, and analysis.
