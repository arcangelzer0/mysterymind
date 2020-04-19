########################################################################

import gridworld as gw
import numpy as np
import cv2

########################################################################
#State Generator Class

class StateGenerator:
    
    def __init__(self, state_size, grid_size):
        
        # Dimensions of the gridworld (height, width)
        self.grid_size = grid_size
        
        # Dimensions of the state array (height, width)
        self.state_size = state_size
        
        # Compute size of flattened state array
        self.flat_size = state_size[0] * state_size[1]
        
    def pad_state(self, state_arr):
        
        # Compute top and bottom padding
        hgt_diff = self.state_size[0] - np.size(state_arr, 0) 
        top_pad = int(np.floor(hgt_diff / 2))
        
        # Compute left and right padding
        wid_diff = self.state_size[1] - np.size(state_arr, 1) 
        lef_pad = int(np.floor(wid_diff / 2))
        
        # Initialize padded array
        state_arr_pad = np.zeros(shape=self.state_size, dtype=np.float)
        
        for r in range(top_pad, np.size(state_arr, 0) + top_pad):
            for c in range(lef_pad, np.size(state_arr, 1) + lef_pad):
                # Insert state_arr values
                state_arr_pad[r][c]= state_arr[r - top_pad][c - lef_pad]
        
        return state_arr_pad
    
    def get_grid_state(self, grid_arr, position):
        
        # Initialize shifted grid state array
        hgt = np.size(grid_arr, 0)
        wid = np.size(grid_arr, 1)
        shift_grid_arr = np.zeros(shape=[hgt, wid], dtype=np.float)
        
        # Get minimum value of the grid array
        g_min = np.min(grid_arr)
        
        # Shift the array from the min and mark the location as zero
        for r in range(np.size(grid_arr, 0)):
            for c in range(np.size(grid_arr, 1)):
                if(r == position[1] and c == position[0]):
                    shift_grid_arr[r][c] = 0.0
                else:
                    shift_grid_arr[r][c] = (grid_arr[r][c] - g_min) + 1
                    
        return shift_grid_arr
    
    def get_field_state(self, curr_grid_arr, prev_grid_arr, position, state_size=[5, 5], scale=1):
        
        # Get dimension of the grid array
        hgt = np.size(curr_grid_arr, 0)
        wid = np.size(curr_grid_arr, 1)
        
        # Initialize field state array
        curr_field_arr = np.zeros(shape=state_size, dtype=np.float)
        prev_field_arr = np.zeros(shape=state_size, dtype=np.float)
        
        # Compute start_row relative to position
        start_row = position[1] - (state_size[0] // 2)
        if(start_row < 0): start_row = 0
        if(start_row + (state_size[0] - 1) > hgt - 1):
            get_exc = (start_row + (state_size[0] - 1)) - (hgt - 1)
            start_row -= get_exc        
        
        # Compute start_col relative to position
        start_col = position[0] - (state_size[1] // 2)
        if(start_col < 0): start_col = 0
        if(start_col + (state_size[1] - 1) > wid - 1):
            get_exc = (start_col + (state_size[1] - 1)) - (wid - 1)
            start_col -= get_exc
        
        # Get field array from the  grid array
        for r in range(start_row, start_row + state_size[0]):
            for c in range(start_col, start_col + state_size[1]):
                curr_field_arr[r - start_row][c - start_col] = curr_grid_arr[r][c]
                prev_field_arr[r - start_row][c - start_col] = prev_grid_arr[r][c]
        
        # Default condition
        if(scale < 1): scale = 1
            
        # Initialize return array
        curr_field_exp = np.zeros(shape=np.dot(scale, state_size), dtype=np.float)
        prev_field_exp = np.zeros(shape=np.dot(scale, state_size), dtype=np.float)
        
        # Expand the state array based on the scale
        for r in range(state_size[0]):
            for c in range(state_size[1]):     
                r_s = scale * r
                c_s = scale * c
                curr_val = curr_field_arr[r][c]
                prev_val = prev_field_arr[r][c]
                for s_r in range(scale):
                    for s_c in range(scale):
                        curr_field_exp[r_s + s_r][c_s + s_c] = curr_val
                        prev_field_exp[r_s + s_r][c_s + s_c] = prev_val
        
        return curr_field_exp, prev_field_exp
    
    def get_location_state(self, start_pnt, end_pnt, position, path_x, path_y, scale=1):
        
        # Initialize location state array
        location_arr = np.zeros(shape=self.grid_size, dtype=np.float)
        
        # Define path as current position with reference to 4 previous positions
        path_len = len(path_x)
        if(path_len < 6): 
            start_path = 1
        else:
            start_path = path_len - 5
        
        # Insert previous path
        for p in range(start_path, len(path_x) - 1):
            location_arr[path_y[p]][path_x[p]] = float(1.0)
            
        # Insert start point
        location_arr[start_pnt[1]][start_pnt[0]] = float(3.0)
        
        # Insert end point
        location_arr[end_pnt[1]][end_pnt[0]] = float(4.0)
        
        # Insert current location
        location_arr[position[1]][position[0]] = float(2.0)
        
                # Default condition
        if(scale < 1): scale = 1
            
        # Initialize return array
        location_arr_exp = np.zeros(shape=np.dot(scale, self.grid_size), dtype=np.float)
                
        # Expand the state array based on the scale
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):     
                r_s = scale * r
                c_s = scale * c
                l_val = location_arr[r][c]
                for s_r in range(scale):
                    for s_c in range(scale):
                        location_arr_exp[r_s + s_r][c_s + s_c] = l_val
            
        return location_arr_exp
    
    def get_energy_state(self, energy, state_size, cell_max=2.0, scale=1):
        
        # Initialize energy state array
        energy_arr = np.zeros(shape=state_size, dtype=np.float)
        
        # Generate energy state array
        input_energy = float(energy)
        r = 0 # Row index
        c = 0 # Column index
        while (input_energy > 0):
            # insert value
            input_energy -= cell_max
            if(input_energy >= 0): energy_arr[r][c] = float(cell_max)
            elif(input_energy > -cell_max): energy_arr[r][c] = float(input_energy + cell_max)
            
            # Append index
            c += 1
            if(c >= state_size[1]):
                c = 0
                r += 1
                
        # Default condition
        if(scale < 1): scale = 1
            
        # Initialize return array
        energy_arr_exp = np.zeros(shape=np.dot(scale, state_size), dtype=np.float)
                
        # Expand the state array based on the scale
        for r in range(state_size[0]):
            for c in range(state_size[1]):     
                r_s = scale * r
                c_s = scale * c
                e_val = energy_arr[r][c]
                for s_r in range(scale):
                    for s_c in range(scale):
                        energy_arr_exp[r_s + s_r][c_s + s_c] = e_val
                
        return energy_arr_exp
    
    def normalize_state(self, state_arr, max_val, min_val):
        
        # Initialize output array
        hgt = np.size(state_arr, 0)
        wid = np.size(state_arr, 1)
        normalize_arr = np.zeros(shape=[hgt, wid], dtype=np.float)
        
        # Iterate through all the elements
        for r in range(hgt):
            for c in range(wid):
                normalize_arr[r][c] = float((state_arr[r][c] - min_val) / (max_val - min_val))
                
        return normalize_arr
    
    def resize_to_imgform(self, input_arr, re_size, def_max=None, def_min=None):
        
        # Get maximum and minimum values
        get_max = np.max(input_arr)
        if(def_max != None): get_max = def_max
        
        get_min = np.min(input_arr)
        if(def_min != None): get_min = def_min
        
        # Normalize input array
        norm_arr = self.normalize_state(input_arr, max_val=get_max, min_val=get_min)
        
        # Initialize image array
        hgt = np.size(input_arr, 0)
        wid = np.size(input_arr, 1)
        img_arr = np.zeros(shape=[hgt, wid], dtype=np.uint8)
        
        # Transform normalized array to uint 8 bit
        for r in range(hgt):
            for c in range(wid):
                img_arr[r][c] = int(255.0 * norm_arr[r][c])
        
        # Using opencv resize function
        resize_img = cv2.resize(img_arr, dsize=(re_size[0], re_size[1]))
        
        return resize_img   
    
    def get_state(self, curr_grid_arr, prev_grid_arr, start_pnt, end_pnt, curr_position, prev_position, path_x, path_y, energy, fstate_size=[5, 5]):
        
        # Get field state
        sft_curr_grid_arr = self.get_grid_state(curr_grid_arr, curr_position)
        sft_prev_grid_arr = self.get_grid_state(prev_grid_arr, prev_position)
        #
        curr_field_arr, prev_field_arr = self.get_field_state(curr_grid_arr=sft_curr_grid_arr, 
                                                              prev_grid_arr=sft_prev_grid_arr,
                                                              position=curr_position,
                                                              state_size=fstate_size,
                                                              scale=4)
        #
        get_gmax = -np.min(curr_grid_arr) + 15
        curr_field_resize = self.resize_to_imgform(curr_field_arr, 
                                                   re_size=self.state_size, 
                                                   def_max=get_gmax, 
                                                   def_min=0) # Convert and resize
        prev_field_resize = self.resize_to_imgform(prev_field_arr, 
                                                   re_size=self.state_size, 
                                                   def_max=get_gmax, 
                                                   def_min=0) # Convert and resize
        #
        curr_field_state = self.normalize_state(curr_field_resize, max_val=255, min_val=0) # Normalize
        prev_field_state = self.normalize_state(prev_field_resize, max_val=255, min_val=0) # Normalize
        
        
        # Get location state
        location_arr = self.get_location_state(start_pnt, end_pnt, curr_position, path_x, path_y, scale=2)
        #
        get_lmax = np.max(location_arr)
        location_arr_resize = self.resize_to_imgform(location_arr, 
                                                     re_size=self.state_size, 
                                                     def_max=get_lmax, 
                                                     def_min = 0)  # Convert and resize
        #
        location_state = self.normalize_state(location_arr_resize, max_val=255, min_val=0) # Normalize
        
        #Define energy array max value per cell
        cell_max = 2
        
        # Get energy state
        energy_arr = self.get_energy_state(energy, state_size=[12, 12], cell_max=cell_max, scale=2)
        #
        energy_arr_resize = self.resize_to_imgform(energy_arr,
                                                   re_size=self.state_size,
                                                   def_max=cell_max,
                                                   def_min=0) # Convert and resize
        #
        energy_state = self.normalize_state(energy_arr_resize, max_val=255, min_val=0)
        
        # Get state array by stacking the curr_field_state, prev_field_state, location_state, and energy_state
        state_array = np.dstack((curr_field_state, prev_field_state, location_state, energy_state))
        
        return state_array

########################################################################
#Reward Function Class 

class RewardFunction:
    
    def __init__(self, pos_max, neg_min):
        
        # Compute positive scale
        self.pos_scale = float(+1.0 / pos_max)
        
        # Compute negative scale
        self.neg_scale = float(-1.0 / neg_min)
        
        # Initialize state transition penalty
        self.delta_s = 0.0
    
    def get_step_reward(self, grid_val):
        
        # Compute scaled grid value reward
        if(grid_val >= 0): 
            scale = self.pos_scale
        else:
            scale = self.neg_scale
    
        # Compute scaled value of the step reward
        grid_val_scaled = grid_val * scale
            
        # Compute step reward
        lambda_mult = 1.0
        step_reward = lambda_mult * grid_val_scaled
        
        return step_reward
    
    def set_delta_s(self, env_delta_s):
        
        self.delta_s = env_delta_s * self.neg_scale
    
    def get_reward(self, grid_val, valid_action, end_episode=0):
        
        # End_Episode: 0 => on-going
        #              1 => success
        #              2 => failed
        
        # Get suitable reward
        reward = float(0.0)
        if(end_episode == 0):         
            if(valid_action):
                reward = self.get_step_reward(grid_val) - self.delta_s
            else:
                # Penalize invalid action
                reward = float(-0.30)
        elif(end_episode == 1):
            reward = float(+0.99)
        elif(end_episode == 2):
            reward = float(-0.99)
            
        # Reward Scale Multiplier
            
        return reward
    
########################################################################
#Environment Class 

class Environment:
    
    def __init__(self, env_id=0, is_default=True, grid_size=[10, 10], state_size=[16, 16]):
        
        # Initialize gridworld environment
        self.gridworld = gw.GridWorld(size=grid_size, default=is_default)
        
        # Initialize gridworld matrix
        self.gridmatrix = self.gridworld.CreateGridWorld()
        
        # Initialize initial energy
        self.initial_energy = 20.0
        
        # Initialize control position
        self.control_position = self.gridworld.GetStartPoint()
        
        # Initialize agent state
        self.agent_state = gw.AgentState(self.control_position[0], self.control_position[1])
        
        # Initialize state size
        self.state_size = state_size
        
        # Initialize state generator
        self.stategenerator = StateGenerator(state_size=state_size, grid_size=grid_size)
        
        # Initialize field state size for get_state
        self.fstate_size = [5, 5]
        
        # initialize previous parameters
        self.prev_grid_arr = self.gridworld.GetCurrentMatrix()
        self.prev_position = self.agent_state.GetCurrentPosition()
        
        # Initialize reward function
        self.rewardfunction = RewardFunction(pos_max=15, neg_min=-25)
        self.rewardfunction.set_delta_s(env_delta_s=self.gridworld.GetDeltaS())
        
        # Get gridworld endpoint
        self.endpoint = self.gridworld.GetEndPoint()
        
        # Initialize episode step count
        self.step_count = 0
        
        # Infinite resource environment parameters
        self.inf_resource = False
        self.p_terminate = -10
        self.max_steps = 200
        
        # Get instatnce id
        self.env_id = env_id
        
    def custom_environment(self, gridmatrix, delta_s, start=None, end=None):
        
        # Set custom gridworld
        self.gridmatrix = self.gridworld.SetCustomGridWorld(gridmatrix=gridmatrix, delta_s=delta_s, start=start, end=end)
        
        # Reinitialize control position
        self.control_position = self.gridworld.GetStartPoint()
        
        # Reinitialize agent state
        self.agent_state = gw.AgentState(self.control_position[0], self.control_position[1])
        
        # Reinitialize state generator
        grid_size = [np.size(self.gridmatrix,0), np.size(self.gridmatrix,1)]
        self.stategenerator = StateGenerator(state_size=self.state_size, grid_size=grid_size)
        
        # Redefine gridworld endpoint
        self.endpoint = self.gridworld.GetEndPoint()
        
        # Redefine state transition penalty
        self.rewardfunction.set_delta_s(env_delta_s=self.gridworld.GetDeltaS())
        
    def set_inf_resource(self, set_env=True, p_terminate=-10, max_steps=None):
        
        # Set infinite resource environment
        self.inf_resource = set_env
        self.p_terminate = p_terminate
        if(max_steps != None):
            self.max_steps = max_steps
        else:
            self.max_steps = int(1.25 * self.gridworld.width * self.gridworld.height)
        
    def reset(self, initial_energy):
        
        # Reset agent state and gridworld environment
        self.control_position = self.gridworld.GetStartPoint()
        self.gridworld.ResetGridWorld()
        self.agent_state.ResetAgentState()
        
        # Set initial energy
        self.initial_energy = initial_energy
        self.gridworld.SetEnergy(self.initial_energy)
        
        # Reset episode step count
        self.step_count = 0
        
        # Get state parameters
        curr_grid_arr = self.gridworld.GetCurrentMatrix()
        self.prev_grid_arr = self.gridworld.GetCurrentMatrix()
        #
        start_pnt = self.gridworld.GetStartPoint()
        end_pnt = self.gridworld.GetEndPoint()
        #
        curr_position = self.agent_state.GetCurrentPosition()
        self.prev_position = self.agent_state.GetCurrentPosition()
        #
        path_x = self.agent_state.GetPathInX()
        path_y = self.agent_state.GetPathInY()
        #
        curr_energy = self.gridworld.GetCurrentEnergy()
        
        # Set reward value max based on initial_energy
        self.rewardfunction.value_max = self.initial_energy + 15
        
        # Generate state input
        state_array = self.stategenerator.get_state(curr_grid_arr=curr_grid_arr,
                                                    prev_grid_arr=self.prev_grid_arr,
                                                    start_pnt=start_pnt,
                                                    end_pnt=end_pnt,
                                                    curr_position=curr_position,
                                                    prev_position=self.prev_position,
                                                    path_x=path_x,
                                                    path_y=path_y,
                                                    energy=curr_energy,
                                                    fstate_size=self.fstate_size)
            
        return state_array
    
    def step(self, action):
        
        # Get action and execute
        get_action = gw.GetNumberAction(action)
        valid_action, new_position, path_value = self.gridworld.ExecuteAction(get_action, self.control_position)
        self.agent_state.ChangePosition(new_position[0], new_position[1], path_value)
        self.control_position = new_position
            
        # Get state parameters
        curr_grid_arr = self.gridworld.GetCurrentMatrix()
        #
        start_pnt = self.gridworld.GetStartPoint()
        end_pnt = self.gridworld.GetEndPoint()
        #
        curr_position = self.agent_state.GetCurrentPosition()
        #
        path_x = self.agent_state.GetPathInX()
        path_y = self.agent_state.GetPathInY()
        #
        curr_energy = self.gridworld.GetCurrentEnergy()
        if(self.inf_resource):
            if(curr_energy <= 0.0): 
                self.gridworld.SetEnergy(0.1)
                curr_energy = 0.1
                self.gridworld.gameover = False
        
        # Generate state input
        state_array = self.stategenerator.get_state(curr_grid_arr=curr_grid_arr,
                                                    prev_grid_arr=self.prev_grid_arr,
                                                    start_pnt=start_pnt,
                                                    end_pnt=end_pnt,
                                                    curr_position=curr_position,
                                                    prev_position=self.prev_position,
                                                    path_x=path_x,
                                                    path_y=path_y,
                                                    energy=curr_energy)
            
        # Update previous parameters
        self.prev_grid_arr = curr_grid_arr.copy()
        self.prev_position = curr_position
        
        # Update episode step count
        self.step_count += 1
        
        if(self.inf_resource):
            if(path_value <= self.p_terminate or self.step_count > self.max_steps): 
                self.gridworld.gameover = True
                self.gridworld.success = False
        
        # Get end condition
        end_condition = 0
        if(self.gridworld.success): end_condition = 1
        elif(self.gridworld.gameover): end_condition = 2            
        
        # Get reward
        path_reward = path_value
        reward = self.rewardfunction.get_reward(grid_val=path_value,
                                                valid_action=valid_action,
                                                end_episode=end_condition)
        
        # Get end episode
        end_episode = False
        if(end_condition > 0): end_episode = True
            
        return state_array, path_reward, reward, end_episode
    
    def render(self, delay=100):
        grid_render = self.gridworld.RenderGridWorld(self.control_position)
        cv2.imshow("Survival Gridworld id=" + str(self.env_id), grid_render)
        cv2.waitKey(delay)
        
    def close_render(self):
        cv2.destroyWindow("Survival Gridworld id=" + str(self.env_id))
        
    def path_render(self, delay=5000):
        grid_size = [self.gridworld.width, self.gridworld.height]
        grid_scale = [self.gridworld.pos_max, self.gridworld.neg_min]
        start = self.gridworld.start
        end = self.gridworld.end
        path_render = self.agent_state.VisualizeAgentPath(grid_size, grid_scale, start, end)
        cv2.imshow("Path Visualization", path_render)
        cv2.waitKey(delay)

    def close_path_render(self):
        cv2.destroyWindow("Path Visualization")
    
    def get_score(self):
        
        return self.gridworld.GetEpisodeScore(self.initial_energy)
    
    def get_success(self):
        
        return self.gridworld.success
    
    def get_gameover(self):
        
        return self.gridworld.gameover
    
        """END OF ENVIRONMENT CLASS"""
