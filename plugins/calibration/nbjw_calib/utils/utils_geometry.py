import numpy as np
import sympy as sp

from scipy.stats import linregress
from ellipse import LsqEllipse


def line_intersection(data, pair, w, h):
    #1e-7 sum in case there are two identical coordinate values
    key1, key2 = pair
    x1, y1, x2, y2 = [], [], [], []
    for count, point in enumerate(data[key1]):
        x1.append(point['x']*w + count*1e-7)
        y1.append(point['y']*h + count*1e-7)
    for count, point in enumerate(data[key2]):
        x2.append(point['x']*w + count*1e-7)
        y2.append(point['y']*h + count*1e-7)
        

    slope1, intercept1, r1, p1, se1 = linregress(x1, y1)
    slope2, intercept2, r2, p2, se2 = linregress(x2, y2)
    
    x_intersection = (intercept2 - intercept1) / (slope1 - slope2)
    y_intersection = slope1 * x_intersection + intercept1

    return x_intersection, y_intersection

def line_polynomial_intersection(x1, y1, x2, y2):
    # Interpolate the two sets of points to create polynomial functions
    if not all(x == x1[0] for x in x1):

        if len(x1) > 2:
            poly1 = np.poly1d(np.polyfit(x1, y1, 2))
            poly2 = np.poly1d(np.polyfit(x2, y2, 1))

            c, b, a = poly1.coeffs
            coef2 = poly2.coeffs
            e, d = coef2 if len(coef2) == 2 else [0, coef2[0]]

            x1 = ((e - b) + np.sqrt((b - e) ** 2 - 4 * c * (a - d))) / (2 * c)
            x2 = ((e - b) - np.sqrt((b - e) ** 2 - 4 * c * (a - d))) / (2 * c)
            y1 = d + e * x1
            y2 = d + e * x2

            return [[x1, y1], [x2, y2]]

        elif len(x1) == 2:
            poly1 = np.poly1d(np.polyfit(x1, y1, 1))
            poly2 = np.poly1d(np.polyfit(x2, y2, 1))

            coef1 = poly1.coeffs
            coef2 = poly2.coeffs

            b, a = coef1 if len(coef1) == 2 else [0, coef1[0]]
            d, c = coef2 if len(coef2) == 2 else [0, coef2[0]]

            x1 = (c - a) / (b - d)
            y1 = a + b * x1

            return [[x1, y1]]
    else:
        return []

def find_ellipse_line_intersections(cx, cy, w, h, theta, a, b):
        x, y = sp.symbols('x y')

        # Equations for the ellipse and the line
        ellipse_eq = ((sp.cos(theta) * (x - cx) - sp.sin(theta) * (y - cy))**2 / w**2 + 
                     (sp.sin(theta) * (x - cx) + sp.cos(theta) * (y - cy))**2 / h**2 - 1)
        line_eq = a + b * x - y

        # Solve the system of equations
        solutions = sp.solve((ellipse_eq, line_eq), (x, y))
        try:
            intersections = [(float(sol[0].evalf()), float(sol[1].evalf())) for sol in solutions]
        except:
            intersections = []
        return intersections


def ellipse_intersection(data, pair, w, h):
    
    key1, key2 = pair
    x1, y1, x2, y2 = [], [], [], []
    
    #Ellipse should be first one of the pair
    for count, point in enumerate(data[key1]):
        x1.append(point['x']*w + count*1e-7)
        y1.append(point['y']*h + count*1e-7)
    for count, point in enumerate(data[key2]):
        x2.append(point['x']*w + count*1e-7)
        y2.append(point['y']*h + count*1e-7)
        
    if len(x1) > 4:
        X = np.array(list(zip(x1, y1)))
            
        reg = LsqEllipse().fit(X)
        
        try:
            center, width, height, phi = reg.as_parameters()
        except:
            return []
        
        if not isinstance(phi, complex):
            coeffs = np.polyfit(x2, y2, 1)
            intersection = find_ellipse_line_intersections(center[0], center[1], width, height, -phi, coeffs[1], coeffs[0])
            
        else:
            intersection = line_polynomial_intersection(x1, y1, x2, y2)
    else:
        intersection = line_polynomial_intersection(x1, y1, x2, y2)
    
    return intersection
    

def find_tangent_points(center, width, height, theta, external_point):
    
    def point_to_ellipse_coords(p, center, theta):
        h, k = center
        x, y = p
        x_1, y_1 = x - h, y - k
        x_2, y_2 = x_1 * np.cos(theta) + y_1 * np.sin(theta), x_1 * np.sin(theta) - y_1 * np.cos(theta)
        
        return x_2, y_2
        
    def feasibility_list(point_list, a, b):
        f_list = []
        smaller, larger = sorted([1-1e-7, 1+1e-7])
        for point in point_list:
            x, y = point
            if smaller <= x**2/a**2 + y**2/b**2 <= larger:
                f_list.append(point)
        return f_list
            
    def ellipse_coords_to_point(p, center, theta):
        x_2, y_2 = p
        h, k = center
        x_1 = x_2 * np.cos(theta) + y_2 * np.sin(theta)
        y_1 = x_2 * np.sin(theta) - y_2 * np.cos(theta)
        x = x_1 + h
        y = y_1 + k
        
        return x, y
    
    # Extract ellipse parameters
    a = width 
    b = height

    px, py = point_to_ellipse_coords(external_point, center, theta)

    y1 = ((a*b)**2*py + np.sqrt(-a**2*b**6*px**2 + a**2*b**4*px**2*py**2 + b**6*px**4)) / (a**2*py**2 + b**2*px**2)
    y2 = ((a*b)**2*py - np.sqrt(-a**2*b**6*px**2 + a**2*b**4*px**2*py**2 + b**6*px**4)) / (a**2*py**2 + b**2*px**2)

    x1_1 = (b*px + np.sqrt(4*a**2*y1*(py-y1) + b**2*px**2)) / (2*b)
    x2_1 = (b*px - np.sqrt(4*a**2*y1*(py-y1) + b**2*px**2)) / (2*b)
    x1_2 = (b*px + np.sqrt(4*a**2*y2*(py-y2) + b**2*px**2)) / (2*b)
    x2_2 = (b*px - np.sqrt(4*a**2*y2*(py-y2) + b**2*px**2)) / (2*b)
    
    points_to_check = [(x1_1, y1), (x2_1, y1), (x1_2, y2), (x2_2, y2)]
    feasible_points = feasibility_list(points_to_check, a, b)

    points_untransformed = [ellipse_coords_to_point(p, center, theta) for p in feasible_points]
    
    return points_untransformed


def are_points_collinear(point1, point2, point3, tolerance=1e-5):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    # Check if the slopes are equal with a tolerance
    return abs((y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1)) < tolerance





                
    
        