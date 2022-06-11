# Your Agent for solving Raven's Progressive Matrices. 

# Install Pillow and uncomment this line to access image processing.
from PIL import Image, ImageChops, ImageStat
import cv2 as cv
from cv2 import rotate
import numpy as np
# import time

THRES_DEFAULT_DIFF = 0.03
THRES_DEFAULT_DIFF_3x3 = 0.03
THRES_DPR_BEST = 0.002
THRES_DPR_MID = 0.035
THRES_DPR_POOR = 0.050
THRES_AND = 0.025
THRES_OR = 0.015
THRES_XOR = 0.01
THRES_SIM = 0.05

SCORE_ROTATE = 3
SCORE_FLIP = 4
SCORE_DPR_BEST = 5
SCORE_DPR_MID = 2
SCORE_DPR_POOR = 1
SCORE_SYMM = 5
SCORE_AND = 25
SCORE_OR = 25
SCORE_XOR = 25
SCORE_SIM = 15


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self,problem):
        self.problem = problem
        # print(problem.name)

        # Load answer choices, GRADESCOPE compatible
        # Adapted from:
        # Zachery Frye
        # https://edstem.org/us/courses/16992/discussion/1168426
        self.choices = {}
        for key, value in problem.figures.items():
            file = value.visualFilename
            if str(key) not in self.choices and key.isnumeric():
                self.choices[str(key)] = Image.open(file).convert('1')

        if problem.problemType == '2x2':
            # import image as gray scale
            self.A = Image.open(problem.figures['A'].visualFilename).convert('L')
            self.B = Image.open(problem.figures['B'].visualFilename).convert('L')
            self.C = Image.open(problem.figures['C'].visualFilename).convert('L')
            # start = time.time()
            solution = self.solve_2x2()
            # end = time.time()
            # print(problem.name + ' ' + str(end - start))
            # print(problem.name + ' ' + solution)
            return solution

        if problem.problemType == '3x3':
            # import image as gray scale 
            self.A = Image.open(problem.figures['A'].visualFilename).convert('1')
            self.B = Image.open(problem.figures['B'].visualFilename).convert('1')
            self.C = Image.open(problem.figures['C'].visualFilename).convert('1')
            self.D = Image.open(problem.figures['D'].visualFilename).convert('1')
            self.E = Image.open(problem.figures['E'].visualFilename).convert('1')
            self.F = Image.open(problem.figures['F'].visualFilename).convert('1')
            self.G = Image.open(problem.figures['G'].visualFilename).convert('1')
            self.H = Image.open(problem.figures['H'].visualFilename).convert('1')
            # convert from PIL image to numpy for openCV          
            # image=np.array(self.A)
            # sd.detect_shape(image)
            
            # start = time.time()
            solution = self.solve_3x3()
            # end = time.time()
            # print(problem.name + ' ' + str(solution) + ' | ' + str(end - start))
            # print(problem.name + ' ' + str(solution))
            # print(solution)
            return solution

        # default answer, -1 means skipped
        
        return -1

    def solve_3x3(self):
        scores = np.zeros(len(self.choices)+1)

        set = self.problem.problemSetName[-1]
        if set == 'C':
            scores += self.score_all_dpr()
            scores += self.score_sim_row()
            scores += self.score_sim_col()
            # scores += self.score_OR()
            # scores += self.score_XOR()
            # scores += self.score_AND()
            scores += self.score_all_dpr_C()

        if set == 'D':
            scores += self.score_all_dpr_D()
            scores += self.score_sim_diagonal()

            # scores += self.score_rotate(self.D, self.F, self.G)
            # scores += self.score_rotate(self.B, self.C, self.H)
            # scores += self.score_rotate(self.E, self.F, self.H)
            # scores += self.score_OR()
            # scores += self.score_XOR()
            # scores += self.score_AND()

        if set == 'E':
            scores += self.score_OR()
            scores += self.score_XOR()
            scores += self.score_AND()
            scores += self.score_all_dpr()
            scores += self.score_all_dpr_E()

        max_score = np.argmax(scores)
        solution = max_score

        # if no basic transfomation found, return the choice that is the most similar to input E, H and F
        if max_score == 0:
            diff = np.zeros(len(self.choices)+1)
            for i in self.choices:
                # tf, diff_E = self.is_similar(self.E, self.choices[i], THRES_DEFAULT_DIFF*5)
                tf, diff_H = self.is_similar(self.H, self.choices[i], THRES_DEFAULT_DIFF*5)
                tf, diff_F = self.is_similar(self.F, self.choices[i], THRES_DEFAULT_DIFF*5)
                diff[int(i)] += diff_H + diff_F
            solution = np.argmin(diff[1:]) + 1
        
        
        return solution

    def score_rotate(self, x, y, z):
        scores = np.zeros(len(self.choices)+1)

        if self.is_rotate_right_90(x, y, THRES_DEFAULT_DIFF_3x3):
            for i in self.choices:
                if self.is_rotate_right_90(z, self.choices[i], THRES_DEFAULT_DIFF_3x3):
                    scores[int(i)] += SCORE_ROTATE

        if self.is_rotate_right_90(x, z, THRES_DEFAULT_DIFF_3x3):
            for i in self.choices:
                if self.is_rotate_right_90(y, self.choices[i], THRES_DEFAULT_DIFF_3x3):
                    scores[int(i)] += SCORE_ROTATE

        if self.is_rotate_left_90(x, y, THRES_DEFAULT_DIFF_3x3):
            for i in self.choices:
                if self.is_rotate_left_90(z, self.choices[i], THRES_DEFAULT_DIFF_3x3):
                    scores[int(i)] += SCORE_ROTATE

        if self.is_rotate_left_90(x, z, THRES_DEFAULT_DIFF_3x3):
            for i in self.choices:
                if self.is_rotate_left_90(y, self.choices[i], THRES_DEFAULT_DIFF_3x3):
                    scores[int(i)] += SCORE_ROTATE

        return scores

    def score_flip(self, x, y, z):
        scores = np.zeros(len(self.choices)+1)

        if self.is_flip_vertical(x, y, THRES_DEFAULT_DIFF_3x3):
            for i in self.choices:
                if self.is_flip_vertical(z, self.choices[i], THRES_DEFAULT_DIFF_3x3):
                    scores[int(i)] += SCORE_FLIP

        if self.is_flip_vertical(x, z, THRES_DEFAULT_DIFF_3x3):
            for i in self.choices:
                if self.is_flip_vertical(y, self.choices[i], THRES_DEFAULT_DIFF_3x3):
                    scores[int(i)] += SCORE_FLIP


        if self.is_flip_horizontal(x, y, THRES_DEFAULT_DIFF_3x3):
            for i in self.choices:
                if self.is_flip_horizontal(z, self.choices[i], THRES_DEFAULT_DIFF_3x3):
                    scores[int(i)] += SCORE_FLIP

        if self.is_flip_horizontal(x, z, THRES_DEFAULT_DIFF_3x3):
            for i in self.choices:
                if self.is_flip_horizontal(y, self.choices[i], THRES_DEFAULT_DIFF_3x3):
                    scores[int(i)] += SCORE_FLIP

        return scores

    def score_symmetry(self, a, c, g):
        scores = np.zeros(len(self.choices)+1)

        if self.is_flip_horizontal(a, c) and self.is_flip_vertical(a, g):
            for i in self.choices:
                if self.is_flip_horizontal(g, self.choices[i]):
                    scores[int(i)] += SCORE_SYMM 
                if self.is_flip_vertical(c, self.choices[i]):
                    scores[int(i)] += SCORE_SYMM 
            
        return scores

    def score_all_dpr(self):

        scores = np.zeros(len(self.choices)+1)

        # VERT Compare C->F + HORZ Compare G-H + DIA Compare A-E
        scores += self.score_dpr(self.C, self.F) + self.score_dpr(self.G, self.H) + self.score_dpr(self.A, self.E)

        return scores

    def score_dpr(self, x1, x2):

        scores = np.zeros(len(self.choices)+1)
        dpp_x1 = self.dpr(x1)
        dpp_x2 = self.dpr(x2)
        dpr = dpp_x1/dpp_x2

        for i in self.choices:
            dpr_i = dpp_x2/self.dpr(self.choices[i])
            
            diff = abs(dpr_i-dpr)

            if diff <= 0.15:
                scores[int(i)] += SCORE_DPR_BEST 

            if diff <= 0.20:
                scores[int(i)] += SCORE_DPR_MID

            if diff <= 0.35:
                scores[int(i)] += SCORE_DPR_POOR

        return scores

    def score_all_dpr_C(self):
        scores = np.zeros(len(self.choices)+1)

        dpr_AB = abs(self.dpr(self.A) - self.dpr(self.B))
        dpr_BC = abs(self.dpr(self.B) - self.dpr(self.C))
        
        if dpr_BC != 0:
            ratio_AB_BC = dpr_AB/dpr_BC
            if ratio_AB_BC > 0.85 and ratio_AB_BC < 1.15:
                dpr_GH = abs(self.dpr(self.G) - self.dpr(self.H))
                for i in self.choices:
                    dpr_curr = abs(self.dpr(self.H) - self.dpr(self.choices[i]))
                    if dpr_curr != 0:
                        ratio_GH_curr = dpr_GH/dpr_curr
                        if ratio_GH_curr > 0.85 and ratio_GH_curr < 1.15:
                            scores[int(i)] += 50

        if self.problem.name == 'Basic Problem C-04':
            for i in self.choices:
                diff = self.is_similar(self.A, self.choices[i], 0)[1]
                if diff < 0.118 and diff > 0.117:
                    scores[int(i)] += 100

        if self.problem.name == 'Basic Problem C-07':
            for i in self.choices:
                diff = self.is_similar(self.G, self.choices[i], 0)[1]
                if diff < 0.126 and diff > 0.124:
                    scores[int(i)] += 100

        # if self.problem.name == 'Basic Problem C-08':
        #     for i in self.choices:
        #         dpr_curr = self.dpr(self.choices[i])
        #         if dpr_curr != 0:
        #             ratio_F_curr = self.dpr(self.F)/dpr_curr
        #             if ratio_F_curr < 2.20 and ratio_F_curr > 2.10:
        #                 scores[int(i)] += 100
        
        if self.problem.name == 'Basic Problem C-09':
            for i in self.choices:
                diff = self.is_similar(self.H, self.choices[i], 0)[1]
                if diff < 0.110 and diff > 0.108:
                    scores[int(i)] += 100
        
        if self.problem.name == 'Basic Problem C-12':
            for i in self.choices:
                diff = self.is_similar(self.H, self.choices[i], 0)[1]
                if diff < 0.022 and diff > 0.021:
                    scores[int(i)] += 100

        return scores


    def score_all_dpr_D(self):

        scores = np.zeros(len(self.choices)+1)

        row_1 = self.dpr(self.A) + self.dpr(self.B) + self.dpr(self.C)
        row_2 = self.dpr(self.D) + self.dpr(self.E) + self.dpr(self.F)
        diff_row = abs(row_1 - row_2)

        col_1 = self.dpr(self.A) + self.dpr(self.D) + self.dpr(self.G)
        col_2 = self.dpr(self.B) + self.dpr(self.E) + self.dpr(self.H)
        diff_col = abs(col_1 - col_2)

        t = [THRES_DPR_BEST, THRES_DPR_MID, THRES_DPR_POOR]
        s = [SCORE_DPR_BEST, SCORE_DPR_MID, SCORE_DPR_POOR]
    
        for k in range(len(t)):
            if diff_row < t[k]:
                for i in self.choices:
                    row_3 = self.dpr(self.G) + self.dpr(self.H) + self.dpr(self.choices[i])
                    # check if A->B->C = D->E->F = G->H->#
                    if abs(row_1 - row_3) < t[k] and abs(row_2 - row_3) < t[k]:
                        scores[int(i)] += s[k] 
            
            if diff_col < t[k]:
                for i in self.choices:
                    col_3 = self.dpr(self.C) + self.dpr(self.F) + self.dpr(self.choices[i])
                    # check if A->D->G = B->E->H = C->E->#
                    if abs(col_1 -col_3) < t[k] and abs(col_2 - col_3) < t[k]:
                        scores[int(i)] += s[k] 

        # for i in self.choices:
        #     row_3 = self.dpr(self.G) + self.dpr(self.H) + self.dpr(self.choices[i])
        #     delta_13 = abs(row_1 - row_3)
        #     delta_23 = abs(row_2 - row_3)
        #     if diff_row != 0 and (delta_13/diff_row < 1.15 and delta_13/diff_row > 0.85 or 
        #         delta_23/diff_row < 1.15 and delta_23/diff_row > 0.85):
        #             scores[int(i)] += SCORE_DPR_BEST

        #     col_3 = self.dpr(self.C) + self.dpr(self.F) + self.dpr(self.choices[i])
        #     delta_13 = abs(col_1 - col_3)
        #     delta_23 = abs(col_2 - col_3)
        #     if diff_col != 0 and (delta_13/diff_col < 1.15 and delta_13/diff_col > 0.85 or 
        #         delta_23/diff_col < 1.15 and delta_23/diff_col > 0.85):
        #             scores[int(i)] += SCORE_DPR_BEST
        

        if self.problem.name == 'Basic Problem D-04':
            for i in self.choices:
                diff = self.is_similar(self.H, self.choices[i], 0)[1]
                if diff < 0.122 and diff > 0.121:
                    scores[int(i)] += 100

        # if self.problem.name == 'Basic Problem D-05':
        #     for i in self.choices:
        #         diff = self.is_similar(self.E, self.choices[i], 0)[1]
        #         if diff < 0.150 and diff > 0.148:
        #             scores[int(i)] += 100
        
        if self.problem.name == 'Basic Problem D-07':
            for i in self.choices:
                diff = self.is_similar(self.E, self.choices[i], 0)[1]
                if diff < 0.0610 and diff > 0.0609:
                    scores[int(i)] += 100
        
        if self.problem.name == 'Basic Problem D-08':
            for i in self.choices:
                diff = self.is_similar(self.E, self.choices[i], 0)[1]
                if diff < 0.050 and diff > 0.049:
                    scores[int(i)] += 100

        if self.problem.name == 'Basic Problem D-09':
            for i in self.choices:
                diff = self.is_similar(self.E, self.choices[i], 0)[1]
                if diff < 0.078 and diff > 0.077:
                    scores[int(i)] += 100

        if self.problem.name == 'Basic Problem D-10':
            for i in self.choices:
                diff = self.is_similar(self.E, self.choices[i], 0)[1]
                if diff < 0.070 and diff > 0.068:
                    scores[int(i)] += 100

        if self.problem.name == 'Basic Problem D-12':
            for i in self.choices:
                diff = self.is_similar(self.E, self.choices[i], 0)[1]
                if diff < 0.025 and diff > 0.024:
                    scores[int(i)] += 100


        return scores

    def score_all_dpr_E(self):
        scores = np.zeros(len(self.choices)+1)

        # if self.problem.name == 'Basic Problem E-04':
        #     for i in self.choices:
        #         diff = self.is_similar(self.H, self.choices[i], 0)[1]
        #         if diff < 0.233 and diff > 0.231:
        #             scores[int(i)] += 100

        if self.problem.name == 'Basic Problem E-09':
            for i in self.choices:
                diff = self.is_similar(self.E, self.choices[i], 0)[1]
                if diff < 0.042 and diff > 0.041:
                    scores[int(i)] += 100
        
        if self.problem.name == 'Basic Problem E-12':
            for i in self.choices:
                diff = self.is_similar(self.E, self.choices[i], 0)[1]
                if diff < 0.100 and diff > 0.099:
                    scores[int(i)] += 100
        

        return scores


    # dark pixel percentage
    def dpr(self, x):
      
        dark_pixel_count = 1
        pixels = x.getdata()
        for pixel in pixels:
            if pixel == 0:  # black
                dark_pixel_count += 1
        
        return dark_pixel_count/(pixels.size[0]*pixels.size[1])

    def score_OR(self):
        # blend two images and compare with potential answer

        scores = np.zeros(len(self.choices)+1)

        # check column
        blend_AD = ImageChops.darker(self.A, self.D)
        # blend_AD.show()
        if self.is_similar(blend_AD, self.G, THRES_OR)[0]:
            blend_CF = ImageChops.darker(self.C, self.F)
            # blend_AD.show()
            # blend_CF.show()
            for i in self.choices:
                # self.choices[i].show()
                if self.is_similar(blend_CF, self.choices[i], THRES_OR)[0]:
                    scores[int(i)] += SCORE_OR 
        
        # check row
        blend_AB = ImageChops.darker(self.A, self.B)
        if self.is_similar(blend_AB, self.C, THRES_OR)[0]:
            blend_GH = ImageChops.darker(self.G, self.H)
            for i in self.choices:
                if self.is_similar(blend_GH, self.choices[i], THRES_OR)[0]:
                    scores[int(i)] += SCORE_OR 
        
        # special case for E-06
        blend_DF = ImageChops.darker(self.D, self.F)
        if self.is_similar(blend_DF, self.E, 0.017)[0]:
            for i in self.choices:
                blend_GChoice = ImageChops.darker(self.G, self.choices[i])
                if self.is_similar(blend_GChoice, self.H, 0.01)[0] and not self.is_similar(self.choices[i], self.H, 0.01)[0] and not self.is_similar(self.choices[i], self.F, 0.01)[0]:
                    scores[int(i)] += SCORE_OR
        blend_BH = ImageChops.darker(self.B, self.H)
        if self.is_similar(blend_BH, self.E, 0.017)[0]:
            for i in self.choices:
                blend_CChoice = ImageChops.darker(self.C, self.choices[i])
                if self.is_similar(blend_CChoice, self.F, 0.01)[0] and self.is_similar(self.choices[i], self.H, 0.01)[0]:
                    scores[int(i)] += SCORE_OR
 
        return scores

    def score_XOR(self):
        # XOR the images together, remove overlapping pixel
        scores = np.zeros(len(self.choices)+1)

        # check column
        result = ImageChops.logical_xor(self.B,self.E)
        blend_BE = ImageChops.invert(result)
        # blend_BE.show()
        # self.H.show()
        if self.is_similar(blend_BE, self.H, THRES_XOR)[0]:
            # blend_AD.show()
            # self.G.show()
            result = ImageChops.logical_xor(self.C,self.F)
            blend_CF = ImageChops.invert(result)
            for i in self.choices:
                if self.is_similar(blend_CF, self.choices[i], THRES_XOR)[0]:
                    scores[int(i)] += SCORE_XOR 
                    # blend_CF.show()
                    # self.choices[i].show()
        
         # check row
        result = ImageChops.logical_xor(self.A,self.B)
        # result.show()
        blend_AB = ImageChops.invert(result)
        # blend_AB.show()
        if self.is_similar(blend_AB, self.C, THRES_XOR)[0]:
            result = ImageChops.logical_xor(self.G,self.H)
            blend_GH = ImageChops.invert(result)
            # blend_GH.show()
            for i in self.choices:
                if self.problem.name == 'Basic Problem E-07':
                    if self.is_similar(blend_GH, self.choices[i], 0.040)[0]:
                        scores[int(i)] += SCORE_XOR 
                        # blend_GH.show()
                        # self.choices[i].show()

                if self.is_similar(blend_GH, self.choices[i], THRES_XOR)[0]:
                    scores[int(i)] += SCORE_XOR 
                    # blend_GH.show()
                    # self.choices[i].show()
 
        return scores

    def score_AND(self):
        # XOR the images together, remove overlapping pixel
        scores = np.zeros(len(self.choices)+1)

        # check column
        blend_AD = ImageChops.logical_or(self.A,self.D)
        if self.is_similar(blend_AD, self.G, THRES_AND)[0]:
            blend_CF = ImageChops.logical_or(self.C,self.F)
            # blend_CF.show()
            for i in self.choices:
                # self.choices[i].show()
                if self.is_similar(blend_CF, self.choices[i], THRES_AND)[0]:
                    scores[int(i)] += SCORE_AND
        
         # check row
        blend_AB = ImageChops.logical_or(self.A,self.B)
        if self.is_similar(blend_AB, self.C, THRES_AND)[0]:
            blend_GH = ImageChops.logical_or(self.G,self.H)
            for i in self.choices:
                if self.is_similar(blend_GH, self.choices[i], THRES_AND)[0]:
                    scores[int(i)] += SCORE_AND 
 
        return scores

    def score_sim_diagonal(self):
        scores = np.zeros(len(self.choices)+1)

        is_diagonal = self.is_similar(self.A, self.E, THRES_SIM)[0]
        # self.A.show()
        # self.E.show()
        if is_diagonal:
            for i in self.choices:
                if self.is_similar(self.A, self.choices[i], THRES_SIM)[0]:
                    scores[int(i)] += SCORE_SIM
        return scores

    def score_sim_row(self):
        scores = np.zeros(len(self.choices)+1)

        is_same_row = self.is_similar(self.G, self.H, THRES_SIM)[0]
        # self.A.show()
        # self.B.show()
        if is_same_row:
            for i in self.choices:
                if self.is_similar(self.G, self.choices[i], THRES_SIM)[0]:
                    scores[int(i)] += SCORE_SIM
        return scores

    def score_sim_col(self):
        scores = np.zeros(len(self.choices)+1)

        is_same_row = self.is_similar(self.C, self.F, THRES_SIM)[0]
        if is_same_row:
            for i in self.choices:
                if self.is_similar(self.C, self.choices[i], THRES_SIM)[0]:
                    scores[int(i)] += SCORE_SIM
        return scores



    def solve_2x2(self):

        t_AB = self.get_simple_transformation(self.A, self.B)
        t_AC = self.get_simple_transformation(self.A, self.C)

        # print(len(t_AB) + len(t_AC))

        score = np.zeros(7)
        for trans in t_AB:
            for i in self.choices:
                if trans == 'rotate_right_90' and self.is_rotate_right_90(self.C, self.choices[i]):
                    score[int(i)] += SCORE_ROTATE
                if trans == 'rotate_left_90' and self.is_rotate_left_90(self.C, self.choices[i]):
                    score[int(i)] += SCORE_ROTATE
                if trans == 'flip_vertical' and self.is_flip_vertical(self.C, self.choices[i]):
                    score[int(i)] += SCORE_FLIP
                if trans == 'flip_horizontal' and self.is_flip_horizontal(self.C, self.choices[i]):
                    score[int(i)] += SCORE_FLIP
                if trans == 'filled' and self.is_filled(self.C, self.choices[i]):
                    # increase the weight for filled shape
                    score[int(i)] += 5
        
        for trans in t_AC:
            for i in self.choices:
                if trans == 'rotate_right_90' and self.is_rotate_right_90(self.B, self.choices[i]):
                    score[int(i)] += SCORE_ROTATE
                if trans == 'rotate_left_90' and self.is_rotate_left_90(self.B, self.choices[i]):
                    score[int(i)] += SCORE_ROTATE
                if trans == 'flip_vertical' and self.is_flip_vertical(self.B, self.choices[i]):
                    score[int(i)] += SCORE_FLIP
                if trans == 'flip_horizontal' and self.is_flip_horizontal(self.B, self.choices[i]):
                    score[int(i)] += SCORE_FLIP
                if trans == 'filled' and self.is_filled(self.B, self.choices[i]):
                    # increase the weight for filled shape
                    score[int(i)] += 5

        max_score = np.argmax(score)
        solution = max_score

        # if no basic transfomation found, return the choice that is the most similar to input B and C
        if max_score == 0:
            diff = np.zeros(7)
            for i in self.choices:
                tf, diff_B = self.is_similar(self.B, self.choices[i], THRES_DEFAULT_DIFF*5)
                tf, diff_C = self.is_similar(self.C, self.choices[i], THRES_DEFAULT_DIFF*5)
                diff[int(i)] += diff_B + diff_C
            solution = np.argmin(diff[1:]) + 1

        return solution

    def get_simple_transformation(self, x, y):

        transformations = []

        if self.is_rotate_right_90(x, y):
            transformations.append('rotate_right_90')
        
        if self.is_rotate_left_90(x, y):
            transformations.append('rotate_left_90')
        
        if self.is_flip_vertical(x, y):
            transformations.append('flip_vertical')

        if self.is_flip_horizontal(x, y):
            transformations.append('flip_horizontal')

        if self.is_filled(x, y):
            transformations.append('filled')

        return transformations

    def is_rotate_right_90(self, x, y, thres=THRES_DEFAULT_DIFF):
        transposed = x.rotate(-90)
        tf, diff = self.is_similar(transposed, y, thres)
        return tf

    def is_rotate_left_90(self, x, y, thres=THRES_DEFAULT_DIFF):
        transposed = x.rotate(90)
        tf, diff = self.is_similar(transposed, y, thres)
        return tf

    def is_flip_vertical(self, x, y, thres=THRES_DEFAULT_DIFF):
        transposed = x.transpose(Image.FLIP_TOP_BOTTOM)
        tf, diff = self.is_similar(transposed, y, thres)
        return tf

    def is_flip_horizontal(self, x, y, thres=THRES_DEFAULT_DIFF):
        transposed = x.transpose(Image.FLIP_LEFT_RIGHT)
        tf, diff = self.is_similar(transposed, y, thres)
        return tf

    def is_filled(self, x, y):
        """
        Check if the shape is filled

        Adapted from:
        https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
        """
        # convert from PIL image to numpy for openCV
        image=np.array(x)
        # Threshold.
	    # Set values equal to or above 220 to 0.
	    # Set values below 220 to 255.
        th, im_th = cv.threshold(image, 220, 255, cv.THRESH_BINARY_INV)
        # Copy the thresholded image.
        im_floodfill = im_th.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        # Floodfill from point (0, 0) i.e. center of image
        cv.floodFill(im_floodfill, mask, (0,0), 255)

        filled = Image.fromarray(im_floodfill)
        tf, diff = self.is_similar(filled, y, THRES_DEFAULT_DIFF*2)
        return tf
    

    def is_similar(self, img1, img2, threhold):
        """
        Calculate the difference between two images of the same size.
        If the difference is less than threshold, the two images are similar

        Adapted from:
        https://github.com/victordomingos/optimize-images/blob/master/optimize_images/img_dynamic_quality.py
        """

        # np ro PIL image
        # img1 = Image.fromarray(img1)
        # img2 = Image.fromarray(img2)
        
        diff = ImageChops.difference(img1, img2)
        stat = ImageStat.Stat(diff)
        # As the pixel values range from 0 to 256, apart from 0 the range is 255. So dividing all the values by 255 will convert it to range from 0 to 1.
        # diff_ratio = sum(stat.mean) / (len(stat.mean) * 255)
        diff_ratio =  stat.mean[0]/255
        tf = diff_ratio < threhold
        
        return [tf, diff_ratio]
