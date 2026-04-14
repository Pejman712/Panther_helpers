# Panther_helpers

The robot must go from **A to B** using:

- **GPS + map route** for the global direction
- **road perception** for local road-centering
- **Nav2** for normal path-following control
- **fusion** to combine both into the final motion command

---

## 1. Global planning

First, the planner computes a path from the start GPS point to the goal GPS point.

### Input
- start point from `/clicked_point`
- goal point from `/clicked_point`
- map / OSM road graph

### Output
- a path topic such as `/plan` or `/gps_route_path`

We can represent the path as:

$$
P = \{p_1, p_2, p_3, \dots, p_n\}
$$

where each path point is:

$$
p_i = (x_i, y_i)
$$

in the `map` frame.

This path answers:

> Which road should the robot follow to reach the destination?

---

## 2. Nav2 baseline control

Nav2 takes the global path and computes a normal driving command.

### Main inputs to Nav2 controller
- path $P$
- current robot pose
- odometry
- costmap / obstacles

### Output
- `/cmd_vel_nav`

This velocity command is:

$$
u_{nav} =
\begin{bmatrix}
v_{nav} \\
\omega_{nav}
\end{bmatrix}
$$

where:
- $v_{nav}$ = linear velocity
- $\omega_{nav}$ = angular velocity

This command means:

> Based only on the planned path, how should the robot move now?

---

## 3. Road perception

The camera image is processed by the road perception node.

### Input
- camera topic, for example `/zed_front/.../image/compressed`

### Processing
The node:
- segments the road
- extracts the road mask
- estimates the visible road center
- estimates the visible road heading
- computes confidence

### Output
- `/road_observation`

The road observation can be simplified as:

$$
z_{road} = \{e_{lat}, e_{head}, c\}
$$

where:
- $e_{lat}$ = lateral road-center error
- $e_{head}$ = road heading error
- $c$ = confidence, with $0 \le c \le 1$

Interpretation:
- if $e_{lat} > 0$, the visible road center is to the right
- if $e_{lat} < 0$, the visible road center is to the left

This observation answers:

> Where is the road in the current camera view?

---

## 4. Fusion logic

The fusion node receives:

- `/cmd_vel_nav`
- `/road_observation`
- `/plan`

and produces the final command:

- `/cmd_vel`

### Inputs
- global route information from `/plan`
- Nav2 control from `/cmd_vel_nav`
- road correction from `/road_observation`

### Road correction
The road-based steering correction can be modeled as:

$$
\omega_{road} = -(k_1 e_{lat} + k_2 e_{head})
$$

where:
- $k_1$ and $k_2$ are gains
- the minus sign makes the robot steer back toward road center

### Blending
The fusion node computes a blending factor:

$$
\alpha = f(c, \text{path curvature}, \text{context})
$$

where:
- $c$ is road confidence
- $\alpha \in [0, 1]$

Then the final steering becomes:

$$
\omega_{final} = \omega_{nav} + \alpha \omega_{road}
$$

and the final linear speed is usually:

$$
v_{final} = \beta v_{nav}
$$

where $\beta$ may be reduced when:
- confidence is low
- the turn ahead is sharp

So the final motion command is:

$$
u_{final} =
\begin{bmatrix}
v_{final} \\
\omega_{final}
\end{bmatrix}
$$

---

## 5. Meaning of the fusion

The key idea is:

- **GPS / route** decides **where to go**
- **road perception** decides **how to stay centered on the visible road**
- **fusion** decides **the final movement**

So the road mask does **not** replace the route.

It only adds a correction.

---

## 6. Behavior in different situations

### Straight road
If the road is visible and confidence is high:

$$
\alpha > 0
$$

so road perception influences steering.

Then:

$$
\omega_{final} \neq \omega_{nav}
$$

This means the robot follows the route while staying better centered on the road.

### Intersection
At an intersection, the camera may see multiple drivable branches.

The route is more important than the road mask, so the fusion should reduce:

$$
\alpha \downarrow
$$

Then:

$$
\omega_{final} \approx \omega_{nav}
$$

This prevents the robot from taking the wrong visible road branch.

### Poor road detection
If the road mask is weak or invalid:

$$
c \approx 0 \Rightarrow \alpha \approx 0
$$

Then the system falls back to pure Nav2:

$$
u_{final} \approx u_{nav}
$$

---

## 7. Topic-level pipeline

```text
/clicked_point
    -> plan_route.py
        -> /plan   (or /gps_route_path)

camera image
    -> road_perception_node.py
        -> /road_observation

Nav2
    -> /cmd_vel_nav

/plan + /road_observation + /cmd_vel_nav
    -> road_nav_fusion_node.py
        -> /cmd_vel
```

---

## 8. Simple intuition

You can think of the system like this:

- the **planner** says:  
  "Take this road sequence to reach the goal."

- **Nav2** says:  
  "To follow that route right now, turn like this."

- the **camera** says:  
  "The visible road center is slightly left/right."

- the **fusion node** says:  
  "I will keep Nav2's route-following command, but I will slightly correct it using the visible road."

---

## 9. Final summary

The full pipeline is:

1. compute a global path from GPS and map data
2. let Nav2 generate a baseline velocity command from that path
3. detect the road in the camera image
4. convert road detection into a small steering correction
5. blend that correction with Nav2's command
6. publish the final `/cmd_vel`

In compact form:

$$
\text{Final command} = \text{Nav2 command} + \text{weighted road correction}
$$

or more explicitly:

$$
\omega_{final} = \omega_{nav} + \alpha \big( -(k_1 e_{lat} + k_2 e_{head}) \big)
$$

This is the core logic of combining GPS navigation and road perception.
