# Problem Definition & Objective

## Industrial Engineering Context

### Background: Automated Optical Inspection (AOI) in Electronics Manufacturing

Printed Circuit Board (PCB) manufacturing is a critical process in electronics production. Quality control is essential because:

- **High Stakes**: A single defective PCB can cause device failure, safety hazards, or costly recalls
- **Volume**: Modern factories produce thousands of boards per day
- **Complexity**: PCBs contain intricate traces, components, and connections that must be perfect
- **Cost**: Catching defects early prevents expensive rework or customer returns

### Current Problem: Manual Inspection Limitations

**Manual Visual Inspection Challenges:**

1. **Speed Bottleneck**
   - Human inspectors: 100-200 boards/hour
   - Production lines: 500-1000 boards/hour
   - Result: Inspection becomes the bottleneck

2. **Error Rates**
   - False Negative Rate: 10-30% (defects missed)
   - False Positive Rate: 5-15% (good boards rejected)
   - Fatigue Factor: Performance degrades after 2-3 hours

3. **Cost Implications**
   - Labor: $40,000-60,000 per inspector annually
   - Missed Defects: $500-5,000 per defective board reaching customers
   - Recalls: $100,000-1,000,000+ for batch recalls

4. **Consistency Issues**
   - Subjective judgment varies between inspectors
   - Performance varies by time of day, fatigue level
   - Training new inspectors takes 3-6 months

### Target Defect Types

The PCB Defects dataset contains six common manufacturing defects:

1. **Mouse Bite**
   - Description: Incomplete routing leaving small gaps
   - Cause: Drill bit or routing tool issues
   - Impact: Weak mechanical connection, potential circuit failure

2. **Open Circuit**
   - Description: Broken or incomplete electrical traces
   - Cause: Etching errors, mechanical damage
   - Impact: Complete circuit failure, device won't function

3. **Short Circuit**
   - Description: Unintended electrical connections between traces
   - Cause: Excess copper, solder bridges
   - Impact: Component damage, fire hazard, device failure

4. **Spurious Copper**
   - Description: Excess copper material where it shouldn't be
   - Cause: Incomplete etching process
   - Impact: Potential shorts, signal interference

5. **Spur**
   - Description: Sharp copper protrusions from traces
   - Cause: Etching irregularities
   - Impact: Risk of shorts, mechanical interference

6. **Missing Hole**
   - Description: Absent or improperly drilled mounting/via holes
   - Cause: Drill bit breakage, positioning errors
   - Impact: Cannot mount components, incomplete connections

### Objective: Deep Learning Solution

**Primary Goal:**
Build an automated defect detection system that achieves:
- **Accuracy**: >95% overall classification accuracy
- **Precision**: >93% (minimize false positives - don't reject good boards)
- **Recall**: >90% (minimize false negatives - catch most defects)
- **Speed**: <50ms inference time per image (real-time capable)

**Business Impact:**
- **Throughput**: Inspect 1000+ boards/hour (10x improvement)
- **Consistency**: 24/7 operation with consistent performance
- **Cost Savings**: Reduce labor costs by 60-80%
- **Quality**: Reduce defect escape rate from 10-30% to <2%
- **ROI**: Payback period of 6-12 months

### Technical Approach

**Model Selection: MobileNetV2**

Why MobileNetV2 is ideal for this application:

1. **Lightweight**: ~14MB model size
   - Deployable on edge devices (Raspberry Pi, NVIDIA Jetson)
   - No need for expensive GPU infrastructure on factory floor

2. **Fast Inference**: 20-50ms per image on CPU
   - Enables real-time inspection on production lines
   - Can process 20-50 boards per second

3. **Efficient Architecture**: Inverted residual blocks
   - Reduces computational cost by 2-3x vs traditional CNNs
   - Lower power consumption for edge deployment

4. **Transfer Learning**: Pretrained on ImageNet
   - Leverages learned features from 1.2M images
   - Requires less PCB-specific training data
   - Faster convergence and better generalization

5. **Industrial Deployment**: Proven in production
   - Used in automotive, aerospace, consumer electronics
   - TensorFlow Lite support for embedded systems
   - Easy integration with factory automation systems

**Data Strategy:**

1. **Augmentation**: Simulate real-world variations
   - Rotation: PCBs may be slightly rotated on conveyor
   - Translation: Camera positioning variations
   - Zoom: Different camera distances
   - Flips: PCBs can be oriented differently

2. **Class Balancing**: Handle imbalanced defect distributions
   - Compute class weights dynamically
   - Ensure rare defects are learned properly
   - Prevent bias toward common defect types

3. **Validation**: Rigorous evaluation
   - 80/20 train/validation split
   - Stratified sampling to maintain class distribution
   - Confusion matrix analysis for per-class performance

### Success Criteria

**Technical Metrics:**
- ✓ Accuracy > 95%
- ✓ Precision > 93% (minimize false alarms)
- ✓ Recall > 90% (catch most defects)
- ✓ F1 Score > 91%
- ✓ Inference time < 50ms

**Business Metrics:**
- ✓ Throughput: 1000+ boards/hour
- ✓ Defect escape rate: <2%
- ✓ False rejection rate: <5%
- ✓ System uptime: >99%
- ✓ ROI: <12 months payback

**Deployment Requirements:**
- ✓ Edge-deployable (no cloud dependency)
- ✓ Real-time capable (production line speed)
- ✓ Robust to lighting variations
- ✓ Easy integration with existing systems
- ✓ Minimal maintenance requirements

### Implementation Phases

**Phase 1: Development** (Current)
- Dataset analysis and preprocessing
- Model training and validation
- Performance optimization
- Documentation and code quality

**Phase 2: Pilot Deployment** (Next)
- Deploy on test production line
- Collect real-world performance data
- Fine-tune model with production images
- Validate against manual inspection

**Phase 3: Production Rollout**
- Deploy across all production lines
- Integrate with factory MES/ERP systems
- Train operators on system usage
- Establish monitoring and retraining pipeline

**Phase 4: Continuous Improvement**
- Collect edge cases and new defect types
- Periodic model retraining
- Performance monitoring and optimization
- Expand to additional defect types

---

## Conclusion

This project addresses a critical industrial need with a practical, deployable solution. By combining state-of-the-art deep learning with industrial engineering requirements, we create a system that:

- Solves real manufacturing problems
- Delivers measurable business value
- Is technically feasible and maintainable
- Can be deployed in production environments

The focus on MobileNetV2 ensures the solution is not just accurate, but also practical for real-world factory deployment where edge computing, real-time performance, and reliability are essential.
