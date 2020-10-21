########################################################################

import numpy as np
import cv2
import random
from enum import Enum

########################################################################
#Enumerable class

class Action(Enum):
    
    Idle  = 0
    Up    = 1
    Down  = 2
    Left  = 3
    Right = 4

########################################################################
#Action Input Function

actionlist = ["w", "s", "a", "d"]

def GetAction(key_input):
    
    action = Action.Idle
    
    if(key_input == actionlist[0]):
        action = Action.Up
        print("Action = UP")
    elif(key_input == actionlist[1]):
        action = Action.Down
        print("Action = DOWN")
    elif(key_input == actionlist[2]):
        action = Action.Left
        print("Action = LEFT")
    elif(key_input == actionlist[3]):
        action = Action.Right
        print("Action = RIGHT")
        
    return action

def GetNumberAction(action_number, print_action=False):
    
    action = Action.Idle
    
    if(action_number == 0):
        action = Action.Up
        if(print_action):
            print("Action = UP")
    elif(action_number == 1):
        action = Action.Down
        if(print_action):
            print("Action = DOWN")
    elif(action_number == 2):
        action = Action.Left
        if(print_action):
            print("Action = LEFT")
    elif(action_number == 3):
        action = Action.Right
        if(print_action):
            print("Action = RIGHT")
        
    return action

########################################################################
#Agent State Class

class AgentState:
    
    def __init__(self, pos_x, pos_y):
        
        #initial position in x
        self.pos_x = pos_x
        self.init_posx = pos_x
        
        #initial position in y
        self.pos_y = pos_y
        self.init_posy = pos_y
        
        #record path in x
        self.path_x = tuple()
        self.path_x += (pos_x,)
        
        #record path in y
        self.path_y = tuple()
        self.path_y += (pos_y,)
        
        #record path rewards and penalties
        self.reward = tuple()
        self.reward += (0,)
        
    def ChangePosition(self, pos_x, pos_y, reward):
        
        #change position in x and update record
        self.pos_x = pos_x
        self.path_x += (pos_x,)
        
        #change position in y and update record
        self.pos_y = pos_y
        self.path_y += (pos_y,)
        
        #update reward records
        self.reward += (reward,)
        
    def ResetAgentState(self):
        
        #reset position in x
        self.pos_x = self.init_posx
        
        #reset position in y
        self.pos_y = self.init_posy
        
        #reset path in x
        self.path_x = tuple()
        self.path_x += (self.init_posx,)
        
        #reset path in y
        self.path_y = tuple()
        self.path_y += (self.init_posy,)
        
        #reset path rewards and penalties
        self.reward = tuple()
        self.reward += (0,)
        
    def GetCurrentPosition(self):      
        return [self.pos_x, self.pos_y]
    
    def GetPathInX(self):
        return self.path_x
    
    def GetPathInY(self):
        return self.path_y
    
    def VisualizeAgentPath(self, size, scale, start, end):
        
        #gridworld cell size
        cel_wid = 30
        cel_hgt = 30
        
        #gridworld minmax value scale
        pos_max = scale[0]
        neg_min = scale[1]
        
        #get matrix row and column count
        wid = cel_wid * size[1]
        hgt = cel_hgt * size[0]
        
        #cell color representation
        color_zero = (80, 208, 146) #light lime green
        color_pstv = (250, 0, 0) #navy blue
        color_ngtv = (0, 0, 200) #fire red
        color_pnts = (2, 166, 249) #gold
        
        #initialize render matrix
        renderpath = np.ndarray(shape=[hgt, wid, 3], dtype=np.uint8)
        renderpath.fill(0)
        
        #render agent path
        for p in range(len(self.reward)):
            #get coordinates and value
            x = cel_wid * self.path_x[p]
            y = cel_hgt * self.path_y[p]
            reward = self.reward[p]
            
            #access matrix value to check previous render
            b_val = renderpath[y + 5, x + 5, 0]
            g_val = renderpath[y + 5, x + 5, 1]
            r_val = renderpath[y + 5, x + 5, 2]
            
            #render path point only once
            if(b_val == 0 and g_val == 0 and r_val == 0):
                #render color
                if(reward != 0):
                    grdt = [0, 0, 0]
                    if(reward > 0):
                        grdt[0] = int(color_zero[0] - (float(color_zero[0] - color_pstv[0])/float(pos_max)) * reward)
                        grdt[1] = int(color_zero[1] - (float(color_zero[1] - color_pstv[1])/float(pos_max)) * reward)
                        grdt[2] = int(color_zero[2] - (float(color_zero[2] - color_pstv[2])/float(pos_max)) * reward)
                    else:
                        grdt[0] = int(color_zero[0] - (float(color_zero[0] - color_ngtv[0])/float(neg_min)) * reward)
                        grdt[1] = int(color_zero[1] - (float(color_zero[1] - color_ngtv[1])/float(neg_min)) * reward)
                        grdt[2] = int(color_zero[2] - (float(color_zero[2] - color_ngtv[2])/float(neg_min)) * reward)
                    cv2.rectangle(renderpath, (x, y), (x + cel_wid, y + cel_hgt), (grdt[0], grdt[1], grdt[2]), cv2.FILLED)    
                else:
                    cv2.rectangle(renderpath, (x, y), (x + cel_wid, y + cel_hgt), color_zero, cv2.FILLED)
                    
                #render values
                x_add = 11
                if(reward != 0):
                    if(reward > 0):
                        if(reward < 10): x_add = 11
                        else: x_add = 6     
                    else:
                        if(reward > -10): x_add = 5 
                        else: x_add = 2 
                cv2.putText(renderpath, str(reward), (x + x_add, y + 18), cv2.FONT_HERSHEY_PLAIN, 0.8, (10, 10, 10), 1)
                
                #render start and end points
                if(self.path_x[p] == start[0] and self.path_y[p] == start[1]):
                    cv2.rectangle(renderpath, (x, y), (x + cel_wid, y + cel_wid), color_pnts, cv2.FILLED)
                    cv2.putText(renderpath, "A", (x + 11, y + 18), cv2.FONT_HERSHEY_PLAIN, 0.8, (10, 10, 10), 1)       
                if(self.path_x[p] == end[0] and self.path_y[p] == end[1]):
                    cv2.rectangle(renderpath, (x, y), (x + cel_wid, y + cel_wid), color_pnts, cv2.FILLED)
                    cv2.putText(renderpath, "B", (x + 11, y + 18), cv2.FONT_HERSHEY_PLAIN, 0.8, (10, 10, 10), 1)
                    
        return renderpath

########################################################################
#Grid World Class

class GridWorld:
    
    def __init__(self, size, default = False):
        
        #width of the gridworld matrix
        self.width = size[1]
        
        #height of the gridworld matrix
        self.height = size[0]
        
        #if defualt is True load defualt matrix if False load random gridworld matrix
        self.default = default
        
        #initialize gridworld matrix
        self.matrix = np.zeros(shape=size, dtype=np.int)
        
        #copy of the original gridworld matrix
        self.matcpy = np.zeros(shape=size, dtype=np.int)
        
        #value of the transition penalty
        self.delta_s = 1.0
        
        #gridworld start position (x, y)
        self.start = [size[1] - 1, size[0] - 1]
        
        #gridworld end position (x, y)
        self.end = [0, 0]
        
        #gridworld maximum positive value
        self.pos_max = 15
        
        #gridworld minimum negative value
        self.neg_min = -25
        
        #game state energy
        self.energy = 5.000
        
        #game state game over
        self.gameover = False
        
        #game state game success
        self.success = False

        #define block cell value
        self.block_val = None
        
    def CreateGridWorld(self):
        
        if (self.default):
            
            #set height and width
            self.height = 10
            self.width = 10
            
            #default matrix
            default_matrix = [[   0,  -1,   0,   5,   0,   8,   0,  -3,  10,   6], 
                              [   0,   8,  -3,   0,   5,  -3, -20, -20,   0,   0],
                              [   4, -20, -20, -20, -20,  10,   0, -20,  -8,   0],
                              [   0,   0,   5,   0,  -3,  -8,  -2,  -8,   0,   3],
                              [   4,  -3,   0,   0,   0,   3,   0,   9,   0,   0],
                              [  -3,   0,  -4,   0,   1,   0,   0,  -2,   5,   0],
                              [   0, -20, -20, -20,  -4,   4, -20, -20, -20, -20],
                              [  -2,   0,   0, -20,   7,   0,   0,   4,   2,   0],
                              [   0,  10,   0,   8,   0,  -4,   0,  -5,   0,   5],
                              [   0,  -5,   0,   0,   1,   0,   5,  -3,  -3,   0]]
            
            #copy the default matrix to the gridworld matrix
            self.matrix = np.zeros(shape=[self.height, self.width], dtype=np.int)
        
            for r in range(self.height):
                for c in range(self.width):
                    self.matrix[r][c] = default_matrix[r][c]
            
            #set start and end points
            self.start = [9, 9]
            self.end = [0, 0]
            
            #set fixed transition penalty
            self.delta_s = 1.2
        else:
            
            if(self.width > 50):
                self.width = 50
                
            if(self.height > 30):
                self.height = 30
                
            #reinitialize gridworld matrix
            self.matrix = np.zeros(shape=[self.height, self.width], dtype=np.int)
            
            #get matrix row and column count
            row_count = int(np.size(self.matrix) / np.size(self.matrix[0]))
            col_count = np.size(self.matrix[0])
            
            #create random gridworld matrix
            for r in range(row_count):
                for c in range(col_count):
                    if(random.randint(0, 1) > 0):
                        if(random.randint(1, 3) == 1):
                            if(random.randint(1, 6) > 4):
                                self.matrix[r][c] = random.randint(6, self.pos_max - 5)
                            else:
                                self.matrix[r][c] = random.randint(1, 5)
                        else:
                            if(random.randint(0, 2) > 1):
                                self.matrix[r][c] = -1 * random.randint(15, -(self.neg_min) - 5) 
                            else:
                                self.matrix[r][c] = -1 * random.randint(1, 14)
                    else:
                        self.matrix[r][c] = 0
            
            #random gridworld start point
            self.start = [random.randint(0, col_count - 1), row_count - 1]
            self.matrix[self.start[1]][self.start[0]] = 0
            
            #random gridworld end point
            self.end = [random.randint(0, col_count - 1), 0]
            self.matrix[self.end[1]][self.end[0]] = 0
            
            #set random transition penalty
            self.delta_s = 0.7 + random.randint(1, 13) / 13.0
            
        #duplicate original copy of the matrix
        self.matcpy = np.zeros(shape=[self.height, self.width], dtype=np.int)
        
        for r in range(self.height):
            for c in range(self.width):
                self.matcpy[r][c] = self.matrix[r][c]
        
        return self.matrix
    
    def SetCustomGridWorld(self, gridmatrix, delta_s, start=None, end=None, block_val=None):
        
        #get dimension of the gridmatrix
        hgt = np.size(gridmatrix, 0)
        wid = np.size(gridmatrix, 1)
        
        if(hgt >= 5 and wid >= 5):    
            #set height and width
            self.height = hgt
            self.width = wid
            
            #get deep copy of the gridmatrix
            self.matrix = np.zeros(shape=[self.height, self.width], dtype=np.int)
            self.matcpy = np.zeros(shape=[self.height, self.width], dtype=np.int)
        
            for r in range(self.height):
                for c in range(self.width):
                    self.matrix[r][c] = gridmatrix[r][c]
                    self.matcpy[r][c] = gridmatrix[r][c]
            
            #set transition penalty
            if(delta_s > 2.0): delta_s = 2.0
            self.delta_s = delta_s
             
            #set start point
            if(start != None): self.start = start   
            else: self.start = [wid - 1, hgt - 1]
                      
            #set end point
            if(end != None): self.end = end    
            else: self.end = [0, 0]
                
            #set block value
            self.block_val = block_val
              
            #set start point value to zero
            self.matrix[self.start[1]][self.start[0]] = 0
        else:  
            #if the dimensions are too small set to defualt instead
            self.default = True
            gridmatrix = self.CreateGridWorld()
            print("Custom gridmatrix is too small initializing to default instead")
            
        return self.matrix
    
    def GetStartPoint(self):
        return [self.start[0], self.start[1]]
    
    def GetEndPoint(self):
        return [self.end[0], self.end[1]]
    
    def GetCurrentMatrix(self):
        return self.matrix
    
    def GetCurrentEnergy(self):
        return self.energy

    def GetDeltaS(self):
        return self.delta_s
    
    def SetEnergy(self, energy):
        self.energy = energy
        
    def SetEndPointValue(self, value):
        
        #clip the value
        if(value < 0): value = 0
        if(value > self.pos_max): value = self.pos_max
        
        #set matrix endpoint value
        self.matrix[self.end[1]][self.end[0]] = value
        
        #set matrix copy endpoint value
        self.matcpy[self.end[1]][self.end[0]] = value
     
    def RenderGridWorld(self, position):
        
        #gridworld cell size
        cel_wid = 30
        cel_hgt = 30
        
        #get matrix row and column count
        row_count = int(np.size(self.matrix)/np.size(self.matrix[0]))
        col_count = np.size(self.matrix[0])
        
        #get render matrix size
        wid = cel_wid * col_count
        hgt = cel_hgt * row_count + cel_hgt
        
        #cell color representation
        color_zero = (80, 208, 146) #light lime green
        color_pstv = (250, 0, 0) #navy blue
        color_ngtv = (0, 0, 200) #fire red
        color_grid = (200, 200, 200) #ash
        color_pnts = (2, 166, 249) #gold
        color_info = (40, 40, 40) #dark gray
        color_bloc = (80, 80, 80) #gray
        
        #initialize render matrix
        render = np.ndarray(shape=[hgt, wid, 3], dtype=np.uint8)
        render.fill(0)
        
        #fill render matrix with color_zero values
        for r in range(cel_hgt, hgt):
            for c in range(wid):
                render[r][c][0] = color_zero[0]
                render[r][c][1] = color_zero[1]
                render[r][c][2] = color_zero[2]
                
        #render header information
        cv2.rectangle(render, (0, 0), (wid, cel_hgt - 1), color_info, cv2.FILLED)
        cv2.line(render, (0, cel_hgt - 1), (wid, cel_hgt - 1), color_grid)
        cv2.putText(render, "E=" + str(self.energy)[0:5], (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, color_grid, 1)
        
        #compute matrix state
        aij = 0
        for r in range(self.height):
            for c in range(self.width):
                if(self.matrix[r][c] != 0): aij += 1
                    
        mat_state = float(aij) / float(self.height * self.width)
        
        agent_loc = ", L=("
        if(position[0] < 10): agent_loc += "0" + str(position[0]) + ", "
        else: agent_loc += str(position[0]) + ","
        if(position[1] < 10): agent_loc += "0" + str(position[1]) + ")"
        else: agent_loc += str(position[1]) + ")"
        
        if(self.width >= 10):
            cv2.putText(render, "N=" + str(mat_state)[0:4] + agent_loc, (wid - 166, 20), cv2.FONT_HERSHEY_PLAIN, 1, color_grid, 1)
    
        x = 0
        y = cel_hgt
        
        #render gridworld
        for r in range(self.height):  
            x = 0
            for c in range(self.width):
                #render color
                if(self.matrix[r][c] != 0):
                    grdt = [0, 0, 0]
                    if(self.matrix[r][c] > 0):
                        grdt[0] = int(color_zero[0] - (float(color_zero[0] - color_pstv[0])/float(self.pos_max)) * self.matrix[r][c])
                        grdt[1] = int(color_zero[1] - (float(color_zero[1] - color_pstv[1])/float(self.pos_max)) * self.matrix[r][c])
                        grdt[2] = int(color_zero[2] - (float(color_zero[2] - color_pstv[2])/float(self.pos_max)) * self.matrix[r][c])
                    else:
                        grdt[0] = int(color_zero[0] - (float(color_zero[0] - color_ngtv[0])/float(self.neg_min)) * self.matrix[r][c])
                        grdt[1] = int(color_zero[1] - (float(color_zero[1] - color_ngtv[1])/float(self.neg_min)) * self.matrix[r][c])
                        grdt[2] = int(color_zero[2] - (float(color_zero[2] - color_ngtv[2])/float(self.neg_min)) * self.matrix[r][c])
                    cv2.rectangle(render, (x, y), (x + cel_wid, y + cel_hgt), (grdt[0], grdt[1], grdt[2]), cv2.FILLED)    
                else:
                    cv2.rectangle(render, (x, y), (x + cel_wid, y + cel_hgt), color_grid)
                    
                #render values
                x_add = 11
                if(self.matrix[r][c] != 0):
                    if(self.matrix[r][c] > 0):
                        if(self.matrix[r][c] < 10): x_add = 11
                        else: x_add = 6     
                    else:
                        if(self.matrix[r][c] > -10): x_add = 5 
                        else: x_add = 2 
                cv2.putText(render, str(self.matrix[r][c]), (x + x_add, y + 18), cv2.FONT_HERSHEY_PLAIN, 0.8, (10, 10, 10), 1)
                
                #render block cells
                if(self.block_val != None):
                    if (self.matrix[r][c] == self.block_val):
                        cv2.rectangle(render, (x, y), (x + cel_wid, y + cel_hgt), color_bloc, cv2.FILLED) 
                
                #render start and end points
                if(r == self.start[1] and c == self.start[0]):
                    cv2.rectangle(render, (x, y), (x + cel_wid, y + cel_wid), color_pnts, cv2.FILLED)
                    cv2.putText(render, "A", (x + 11, y + 18), cv2.FONT_HERSHEY_PLAIN, 0.8, (10, 10, 10), 1)       
                if(r == self.end[1] and c == self.end[0]):
                    cv2.rectangle(render, (x, y), (x + cel_wid, y + cel_wid), color_pnts, cv2.FILLED)
                    cv2.putText(render, "B", (x + 11, y + 18), cv2.FONT_HERSHEY_PLAIN, 0.8, (10, 10, 10), 1)
                    
                #render agent position
                if(r == position[1] and c == position[0]):
                    cv2.circle(render, (x + int(cel_wid/2), y + int(cel_hgt/2)), 12, (200, 25, 190), cv2.FILLED)
                
                x += cel_wid
            y += cel_hgt
                
        return render
                
    def ExecuteAction(self, action, position):
        
        #update agent position
        new_position = position.copy()
        
        #check valid action
        valid_action = False
        
        #validate action and update position
        if(action == Action.Up):
            if(position[1] - 1 > -1):
                new_position[1] -= 1
                valid_action = True
        elif(action == Action.Down):
            if(position[1] + 1 < self.height):
                new_position[1] += 1
                valid_action = True
        elif(action == Action.Left):
            if(position[0] - 1 > -1):
                new_position[0] -= 1
                valid_action = True
        elif(action == Action.Right):
            if(position[0] + 1 < self.width):
                new_position[0] += 1
                valid_action = True
               
        #validate action on block cell
        if(self.block_val != None):
            if(self.matrix[new_position[1]][new_position[0]] == self.block_val):
                new_position = position
                valid_action = False
                
        #initialize path value
        path_value = 0
        
        if(valid_action):
            #get matrix value from the current position
            path_value = self.matrix[new_position[1]][new_position[0]]
            
            #update matrix if value is not zero
            if(path_value != 0):
                #change value to zero if the condition is satisfied
                self.matrix[new_position[1]][new_position[0]] = 0
                
        #update energy from the path_value and transition penalty
        self.energy += path_value - self.delta_s
        if(self.energy < 0.01): self.energy = 0.0
        
        #game over condition
        if(self.energy <= 0.01):
            self.gameover = True
            self.success = False
            
        #game success condition
        if(new_position[0] == self.end[0] and new_position[1] == self.end[1]):
            self.success = True
            self.gameover = False
                
        return valid_action, new_position, path_value

    def GetEpisodeScore(self, initial_energy):
        
        #initialize return variable
        score = 0.0
        
        #compute episode score
        if(self.success):
            score = np.clip(self.energy - initial_energy + 1.0, a_min=1.0, a_max=None)
        
        return score
    
    def ResetGridWorld(self):
        
        #get copy of the original gridworld matrix
        self.matrix = np.zeros(shape=[self.height, self.width], dtype=np.int)
        
        for r in range(self.height):
            for c in range(self.width):
                self.matrix[r][c] = self.matcpy[r][c]
        
        #reset game status
        self.gameover = False
        self.success = False
        self.energy = 5.0
        
    def GetLocationState(self, position):
        
        #initialize location matrix
        loc_matrix = np.zeros(shape=[self.height, self.width], dtype=np.float16)
        
        #define location matrix values
        for r in range(self.height):
            for c in range(self.width):
                if(r == position[1] and c == position[0]): loc_matrix[r][c] += self.energy
                elif(r == self.start[1] and c == self.start[0]): loc_matrix[r][c] += -10.0
                elif(r == self.end[1] and c == self.end[0]): loc_matrix[r][c] += 10.0
                    
        return loc_matrix

########################################################################
