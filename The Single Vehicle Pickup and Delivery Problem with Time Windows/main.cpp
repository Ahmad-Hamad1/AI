#include <bits/stdc++.h>

using namespace std;

int locations;
double currentTime = 0;
int content = 0;
int capacity;

map<int, int> haveDemand;
set<int> toVisit;
stack<int> st;


typedef struct task {
    double pickedAt;
    double deliveredAt;
    int destination;
    int source;
    int start;
    int end;
    int id;

    task(int id, int source, int destination, int start, int end) {
        this->id = id;
        this->source = source;
        this->destination = destination;
        this->start = start;
        this->end = end;
        this->pickedAt = -1;
        this->deliveredAt = -1;
    }
} task;

typedef struct node {
    bool pickup, delivery;
    double x, y;
    int id;
    int start, end;
    int taskD;
    int taskP;

    node(double x, double y, int id) {
        this->pickup = this->delivery = false;
        this->x = x;
        this->y = y;
        this->id = id;
        this->start = -1;
        this->end = -1;
        this->taskD = -1;
        this->taskP = -1;
    }

    void setStart(int start) {
        this->start = start;
    }

    void setEnd(int end) {
        this->end = end;
    }

    void setTaskD(int task) {
        this->taskD = task;
    }

    void setTaskP(int task) {
        this->taskP = task;
    }
} node;


bool done(vector<bool> vis) {
    for (auto item : toVisit)
        if (!vis[item])
            return false;
    return true;
}


void MRV(int current, vector<bool> vis, vector<node> nodes, queue<pair<int, int>> &q) {
    multiset<pair<int, int>> ml; // initializing the multiset
    for (int i = 0; i < nodes.size(); i++) {  // iterating through all locations
        if (vis[i] || i == current)  // skipping the current location and the already visited ones
            continue;
        double distance = sqrt(pow(nodes[i].x - nodes[current].x, 2) + pow(nodes[i].y - nodes[current].y,
                                                                           2)); // calculating the distance from the current location
        double timeConstraint =
                nodes[i].end - (distance + currentTime);  // calculating the time constrain (the heuristic)
        if (timeConstraint >=
            0) {                                        // checking if the time constrain is within the time window
            if ((nodes[i].pickup && !nodes[i].delivery && content == capacity) ||  // checking the capacity constrain
                (nodes[i].delivery && !vis[haveDemand[i]]))
                continue;
            ml.insert({timeConstraint,
                       i});      // adding the location to the multi set if it satisfies the time and capacity constrains
        }
    }
    for (auto i : ml) {
        q.push({sqrt(pow(nodes[i.second].x - nodes[current].x, 2) + pow(nodes[i.second].y - nodes[current].y,
                                                                        2)), // adding the final available node to the queue with their distances
                i.second});
    }
}

bool backTrack(int v, vector<bool> &vis, vector<node> nodes, vector<task> &tasks, double prevTime) {
    vis[0] = true;  // marking the depot location as visited
    queue<pair<int, int>> q;  // initializing the queue
    MRV(v, vis, nodes, q); // calling the back track function
    while (!q.empty()) { // iterating through all the locations available in order depending on the heuristic
        vis[(q.front()).second] = true;  // marking the location as visited
        double prevTime = currentTime;    // storing the time until now to recover it if we reach a dead end from this location
        currentTime + q.front().first >= (double) nodes[q.front().second].start ? currentTime += q.front().first
                                                                                : currentTime = (double) nodes[q.front().second].start; // adding the required time to reach the location to the current time
        if (nodes[(q.front()).second].pickup) {  // taking the demand from the node if it as a pick up location
            content++;                           // adding the demand to the current load
            tasks[nodes[(q.front()).second].taskP].pickedAt = currentTime; // storing the time in which the pick up is done
        }
        if (nodes[(q.front()).second].delivery) {                          // delivering the demand if the location is a delivery one
            content--;                                                     // removing the demand from the current load
            tasks[nodes[(q.front()).second].taskD].deliveredAt = currentTime; // storing the time in which the delivery is done
        }
        backTrack((q.front()).second, vis, nodes, tasks,
                  prevTime); // calling the backtrack function again for the current node
        if (done(vis))    // checking if all the locations are visited in order to end backtracking
            return true;
        q.pop(); // removing the node from the queue
    }
    if (done(vis)) { // checking if all the locations are visited in order to end backtracking
        return true;
    }
    // otherwise we reach a dead end
    currentTime = prevTime;// returning the time to the lst one before the current call
    if (nodes[v].pickup)   // returning the taken demand if the location is a pick up one
        content--;
    else if (nodes[v].delivery)  // re-taking the demand if the location is a delivery one
        content++;
    vis[v] = false; // marking the location as not visited again
    return false; // returning false indicating that we have reached a dead end
}

int main() {
    ios::sync_with_stdio(false);
    cout.tie(nullptr);
    cin.tie(nullptr);
    FILE *coordinates = fopen("coordinates.txt", "r"), *tasksFile = fopen("tasks.txt", "r");
    // Reading Coordinates File.
    fscanf(coordinates, "%d", &locations); // First read number of locations from coordinates file.
    vector<node> loc;
    double x, y;
    int id = 0; // To use as an ID for each location, after adding a location it will be incremented.
    while (fscanf(coordinates, "%lf%lf", &x, &y) != EOF) {
        loc.push_back(node(x, y,
                           id)); // Add a node contains information (Its ID and X, Y coordinated) about read location to the locations vector (loc).
        id++; // Increment the ID for next location.
    }
    // Reading Tasks File.
    vector<task> tasks;
    fscanf(tasksFile, "%d", &capacity); // First read the capacity of the car from tasks file.
    int start, end, source, destination;
    id = 0; // Now its used to add ID's for each read task.
    while (fscanf(tasksFile, "%d%d%d%d", &source, &destination, &start, &end) != EOF) {
        tasks.push_back(task(id, source, destination, start, end)); // Add each task to a vector of tasks.
        loc[source].pickup = true; // Mark the pickup location in loc vector of the task as pickup.
        loc[destination].delivery = true; // Mark the delivery location in loc vector of the task as delivery.
        // ---------------------------------------------------------------------------------------------
        // Specify the start and end times for the source and delivery locations of the read task
        loc[source].setStart(start);
        loc[destination].setStart(start);
        loc[source].setEnd(end);
        loc[destination].setEnd(end);
        //----------------------------------------------------------------------------------------------
        // Specify in which task each location in involved so we can easily know the location is pickup or delivery for which task.
        loc[source].setTaskP(id);
        loc[destination].setTaskD(id);
        //----------------------------------------------------------------------------------------------
        haveDemand[destination] = source; // Map the source for each destination in each task so we can know from where does the pickup come for each destination.
        //----------------------------------------------------------------------------------------------
        // Mark the source and destination for each task as must be visited.
        toVisit.insert(source);
        toVisit.insert(destination);
        //-----------------------------------------------------------------------------------------------
        id++; // Increment ID for next task.
    }
    vector<bool> vis(locations); // Marks each location as visited or not.
    st.push(0);
    bool solution = backTrack(0, vis, loc, tasks, 0);
    if (solution) {
        for (int i = 0; i < tasks.size(); i++)
            cout << "Task : " << i << ", With time window [" << tasks[i].start << ", " << tasks[i].end
                 << "], Delivered from " << tasks[i].source << " to " << tasks[i].destination
                 << ", Picked up at : " << tasks[i].pickedAt << ", Delivered  at : " << tasks[i].deliveredAt << endl;
    } else
        cout << "No Solution Available !!" << endl;

    fclose(coordinates);
    fclose(tasksFile);

    return 0;
}