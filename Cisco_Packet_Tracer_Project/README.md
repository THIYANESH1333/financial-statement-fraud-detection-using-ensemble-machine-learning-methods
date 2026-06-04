# Cisco Packet Tracer Network Topology Project

## Project Overview
This project recreates a complex network topology with three major cities (Delhi, Mumbai, Chennai) connected through multiple paths with different routing metrics. The network demonstrates both RIP and OSPF routing protocols.

## Network Topology
- **3 Main Cities**: Delhi, Mumbai (Destination), Chennai (Source)
- **6 Intermediate Nodes**: HOP 1, HOP 2, HOP 3, HOP 4, HOP 5, HOP 6
- **Multiple Paths**: Direct connections and multi-hop paths
- **Routing Protocols**: RIP and OSPF with different cost metrics

## How to Create the Project in Packet Tracer

### Step 1: Open Cisco Packet Tracer
1. Launch Cisco Packet Tracer
2. Click `File` → `New` → `Blank Project`
3. Save as `Network_Topology.pkt`

### Step 2: Add Devices
**Main City Routers (3x):**
- Drag 3x Router 2811 to workspace
- Position: Delhi (top-center), Mumbai (bottom-left), Chennai (bottom-right)

**Intermediate Routers (6x):**
- Drag 6x Switch 2960 to workspace
- Position: HOP1-HOP6 between the main cities

### Step 3: Connect Devices
**Direct Links:**
- Delhi Fa0/0 ↔ Mumbai Fa0/0 (Cost: 100)
- Delhi Fa0/1 ↔ Chennai Fa0/0 (Cost: 100)

**Multi-hop Path 1 (Chennai → Mumbai):**
- Chennai Fa0/1 ↔ HOP1 Fa0/0 (Cost: 10, BW: 200)
- HOP1 Fa0/1 ↔ HOP2 Fa0/0 (Cost: 10, BW: 20)
- HOP2 Fa0/1 ↔ Mumbai Fa0/1 (Cost: 10)

**Multi-hop Path 2 (Delhi → Chennai):**
- Delhi Fa1/0 ↔ HOP3 Fa0/0 (Cost: 20)
- HOP3 Fa0/1 ↔ HOP4 Fa0/0 (Cost: 20)
- HOP4 Fa0/1 ↔ HOP5 Fa0/0 (Cost: 20)
- HOP5 Fa0/1 ↔ HOP6 Fa0/0 (Cost: 10)
- HOP6 Fa0/1 ↔ Chennai Fa1/0 (Cost: 10, BW: 170)

### Step 4: Apply Configurations
Copy and paste the configurations from the Configurations folder into each router's CLI.

### Step 5: Test Network
```bash
# From Delhi router
ping 192.168.1.2  # Test Delhi to Mumbai
ping 192.168.2.2  # Test Delhi to Chennai
```

## File Structure
```
Cisco_Packet_Tracer_Project/
├── README.md
├── Configurations/
│   ├── Delhi_Router.txt
│   ├── Mumbai_Router.txt
│   ├── Chennai_Router.txt
│   └── HOP1-HOP6_Router.txt
├── Documentation/
│   ├── Setup_Guide.md
│   └── Configuration_Guide.md
└── Scripts/
    ├── test_connectivity.py
    └── network_analysis.py
```

## Quick Start
1. Open Cisco Packet Tracer
2. Create new project
3. Add devices as described above
4. Apply configurations from Configurations folder
5. Test connectivity

## Features
- OSPF routing with custom cost metrics
- RIP routing protocol implementation
- Multiple redundant paths
- Bandwidth and hop count specifications
- Comprehensive testing and verification tools
