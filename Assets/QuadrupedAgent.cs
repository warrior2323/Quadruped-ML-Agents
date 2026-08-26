using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class QuadrupedAgent : Agent
{
    [Header("Body Parts")]
    public ArticulationBody mainBody;
    public ArticulationBody[] legs; // Array holding all 8 leg parts

    [Header("Environment")]
    public Transform targetBox;
    private Vector3 startingPosition;
    private Quaternion startingRotation; // Save the horizontal rotation

    public override void Initialize()
    {
        // Save the exact spot the dog spawns so we can teleport it back later
        startingPosition = mainBody.transform.position;
        startingRotation = mainBody.transform.rotation; 
        
        // Force the episode to end after 3000 steps
        MaxStep = 3000; 
    }

    public override void OnEpisodeBegin()
    {
        // 1. RESET THE DOG
        // Teleport the dog back to the center and kill all its momentum
        mainBody.TeleportRoot(startingPosition, startingRotation);
        mainBody.linearVelocity = Vector3.zero;
        mainBody.angularVelocity = Vector3.zero;

        // Safely reset the motors to 0 so they don't violently kick on spawn
        for (int i = 0; i < legs.Length; i++)
        {
            var drive = legs[i].xDrive;
            drive.target = 0f;
            legs[i].xDrive = drive;
        }

        // 2. RESET THE TARGET
        // Move the TargetBox to a random spot within a 10-meter radius
        float randomX = Random.Range(-15f, 15f);
        float randomZ = Random.Range(-15f, 15f);
        targetBox.position = new Vector3(startingPosition.x + randomX, 0.5f, startingPosition.z + randomZ);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Tell the neural net where the dog is looking and where the box is
        Vector3 directionToTarget = targetBox.position - mainBody.transform.position;
        
        // Feed the data into the neural network (The Observation Space: 23 total)
        sensor.AddObservation(directionToTarget.normalized); // 3 values (X,Y,Z)
        sensor.AddObservation(mainBody.transform.forward);   // 3 values (Which way is the dog facing?)
        
        sensor.AddObservation(mainBody.linearVelocity);      // 3 values (How fast is it moving?)
        sensor.AddObservation(mainBody.angularVelocity);     // 3 values (Is it spinning/falling?)
        sensor.AddObservation(mainBody.transform.up);        // 3 values (Which way is "Up" for the dog?)
        
        foreach (var leg in legs)
        {
            sensor.AddObservation(leg.jointPosition[0]); // 8 values total (1 for each leg)
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // 1. MOVE THE MOTORS
        for (int i = 0; i < legs.Length; i++)
        {
            var drive = legs[i].xDrive;
            float rawAction = actions.ContinuousActions[i];
            float normalizedAction = (rawAction + 1f) / 2f;
            drive.target = Mathf.Lerp(drive.lowerLimit, drive.upperLimit, normalizedAction);
            legs[i].xDrive = drive;
        }

        // 2. DENSE REWARD SHAPING
        
        Vector3 directionToTarget = (targetBox.position - mainBody.transform.position).normalized;
        float forwardSpeed = Vector3.Dot(mainBody.linearVelocity, directionToTarget);
        
        AddReward(forwardSpeed * 0.1f); 

        float energyUsed = 0f;
        for (int i = 0; i < actions.ContinuousActions.Length; i++)
        {
            energyUsed += Mathf.Pow(actions.ContinuousActions[i], 2); 
        }
        AddReward(-energyUsed * 0.005f);

        // If the dog jumps higher than 1.3 meters, punish it every single frame it is in the air.
        if (mainBody.transform.position.y > 2.25f) 
        {
            AddReward(-0.05f);
        }


        // 3. EPISODE TERMINATION (The Final Score)

        float distanceToTarget = Vector3.Distance(mainBody.transform.position, targetBox.position);

        if (distanceToTarget < 1.5f)
        {
            SetReward(1.0f); // Massive bonus for succeeding
            EndEpisode(); 
        }

        // Now it ONLY dies if its center hits the floor (y < 1.2f).
        else if (mainBody.transform.position.y < 1.2f)
        {
            SetReward(-1.0f); // Massive penalty for face-planting
            EndEpisode();
        }
    }

    // 4. COLLISION DETECTION

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.name.Contains("wall"))
        {
            SetReward(-1.0f); // Give a massive penalty for crashing
            EndEpisode();     // Instantly reset the run
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;

        for (int i = 0; i < 8; i++)
        {
            continuousActionsOut[i] = 0f;
        }

        if (Input.GetKey(KeyCode.W)){ 
            continuousActionsOut[0] = 1f;
            Debug.Log("W Key Pressed - Target is 1");
        }
        if (Input.GetKey(KeyCode.S)) continuousActionsOut[0] = -1f;
        if (Input.GetKey(KeyCode.E)) continuousActionsOut[1] = 1f;
        if (Input.GetKey(KeyCode.D)) continuousActionsOut[1] = -1f;
        if (Input.GetKey(KeyCode.R)) continuousActionsOut[2] = 1f;
        if (Input.GetKey(KeyCode.F)) continuousActionsOut[2] = -1f;
        if (Input.GetKey(KeyCode.T)) continuousActionsOut[3] = 1f;
        if (Input.GetKey(KeyCode.G)) continuousActionsOut[3] = -1f;
        if (Input.GetKey(KeyCode.Y)) continuousActionsOut[4] = 1f;
        if (Input.GetKey(KeyCode.H)) continuousActionsOut[4] = -1f;
    }
}