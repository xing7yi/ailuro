from pathlib import Path
import gmsh

def create_hertz_model(file_name:str):
    """
    创建一个Hertz接触模型的网格文件。
    """
    # 使用Gmsh OCC API创建Hertz模型
    gmsh.initialize()
    gmsh.model.add("Hertz_Model")
    f = gmsh.model.occ
    # 定义Hertz模型的几何

    #  geometry
    #  ┌───────────────6───────────────┐
    #  7                               5
    #  └───────────────4───────────────┘
    #
    #  
    #  * *
    #  *       *
    #  *            *
    #  *               1 
    #  *                 *  
    #  2                   *
    #  *                     *
    #  *                      *
    #  *                       *
    #  *   *   *   3   *   *   *
    # 
    # 先创建半圆
    # 定义圆心和半径
    radius = 1.0

    # 定义点
    p_center = f.addPoint(0, 0, 0)    
    p1 = f.addPoint(0, radius, 0, 0.01)
    p2 = f.addPoint(radius, 0, 0, 0.01)
    # 定义线和圆弧
    arc = f.addCircleArc(p1, p_center, p2) # 创建上半圆弧
    line_y = f.addLine(p_center, p1)
    line_x = f.addLine(p2, p_center)  # 创建水平线
    # 创建曲线环和面
    cloop = f.addCurveLoop([line_y, arc, line_x])
    specimen_surface = f.addPlaneSurface([cloop])

    # 定义一个长方形
    width = 1.2  # 长方形的宽度
    height = 0.05  # 长方形的高度
    div_num_w = int(width / 0.01)
    div_num_h = int(height / 0.02)
    y_offset = 1*height  # 长方形的y偏移量
    f.addRectangle(0, radius + y_offset, 0, width, height)  # 定义一个长方形

    f.synchronize()

    # 定义物理组
    # 定义半圆的物理组
    body_group_specimen = gmsh.model.addPhysicalGroup(2, [specimen_surface])
    gmsh.model.setPhysicalName(2, body_group_specimen, "Specimen_Body")

    # 定义长方形的物理组
    body_group_indenter = gmsh.model.addPhysicalGroup(2, [2])
    gmsh.model.setPhysicalName(2, body_group_indenter, "Indenter_Body")

    # 定义1D物理组 (边界)

    bc_edge_group_dict = {
        "Indenter_Top_Edge": [6],
        "Indenter_Left_Edge": [7],
        "Indenter_Bottom_Edge": [4],
        "Specimen_Top_Edge": [1],
        "Specimen_Left_Edge": [2],
        "Specimen_Bottom_Edge": [3]
    }
    for edge_name, tags in bc_edge_group_dict.items():
        group = gmsh.model.addPhysicalGroup(1, tags)
        gmsh.model.setPhysicalName(1, group, edge_name)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.01)  # 设置最小网格大小
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)  # 设置最大网格大小

    # gmsh.model.mesh.setSize([(1,arc)],0.01)
    gmsh.model.mesh.setTransfiniteCurve(6, div_num_w+1)
    gmsh.model.mesh.setTransfiniteCurve(4, div_num_w+1)
    gmsh.model.mesh.setTransfiniteCurve(7, div_num_h+1)
    gmsh.model.mesh.setTransfiniteCurve(5, div_num_h+1)

    # 生成结构网格
    # gmsh.model.mesh.setTransfiniteSurface(1, "Alternate")  # 设置跨限算法
    gmsh.model.mesh.setTransfiniteSurface(2, "Alternate")  # 设置跨限算法
    # gmsh.model.mesh.setRecombine(2, 1)  # 重组为四边形网格
    gmsh.model.mesh.setRecombine(2, 2)  # 重组为四边形网格
    gmsh.model.mesh.generate(2)  # 生成2D网格

    # 保存文件
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(file_name)
    print(f"Gmsh OCC模型已成功写入文件: {file_name}")

    gmsh.fltk.run()

    gmsh.finalize()

    # 启动gmsh
    
def create_hertz_model_contact_area_refine(file_name:str):
    """
    创建一个Hertz接触模型的网格文件。
    """
    # 使用Gmsh OCC API创建Hertz模型
    gmsh.initialize()
    gmsh.model.add("Hertz_Model")
    occ = gmsh.model.occ
    # 定义Hertz模型的几何

    #  geometry
    #  ┌───────────────6───────────────┐
    #  7                               5
    #  └───────────────4───────────────┘
    #
    #  
    #  * *
    #  *       *
    #  *            *
    #  *               1 
    #  *                 *  
    #  2                   *
    #  *                     *
    #  *                      *
    #  *                       *
    #  *   *   *   3   *   *   *
    # 
    # 先创建半圆
    # 定义圆心和半径
    radius = 1.0

    # 定义点
    p_center = occ.addPoint(0, 0, 0)    
    p1 = occ.addPoint(0, radius, 0)
    p2 = occ.addPoint(radius, 0, 0)
    # 定义线和圆弧
    arc = occ.addCircleArc(p1, p_center, p2) # 创建上半圆弧
    line_y = occ.addLine(p_center, p1)
    line_x = occ.addLine(p2, p_center)  # 创建水平线
    # 创建曲线环和面
    cloop = occ.addCurveLoop([line_y, arc, line_x])
    specimen_surface = occ.addPlaneSurface([cloop])

    # 定义一个长方形
    width = 1.2  # 长方形的宽度
    height = 0.1  # 长方形的高度
    div_num_w = int(width / 0.05)
    div_num_h = int(height / 0.05)
    y_offset = 1*height  # 长方形的y偏移量
    occ.addRectangle(0, radius + y_offset, 0, width, height)  # 定义一个长方形

    occ.synchronize()

    # 定义物理组
    # 定义半圆的物理组
    body_group_specimen = gmsh.model.addPhysicalGroup(2, [specimen_surface])
    gmsh.model.setPhysicalName(2, body_group_specimen, "Specimen_Body")

    # 定义长方形的物理组
    body_group_indenter = gmsh.model.addPhysicalGroup(2, [2])
    gmsh.model.setPhysicalName(2, body_group_indenter, "Indenter_Body")

    # 定义1D物理组 (边界)

    bc_edge_group_dict = {
        "Indenter_Top_Edge": [6],
        "Indenter_Left_Edge": [7],
        "Indenter_Bottom_Edge": [4],
        "Specimen_Top_Edge": [1],
        "Specimen_Left_Edge": [2],
        "Specimen_Bottom_Edge": [3]
    }
    for edge_name, tags in bc_edge_group_dict.items():
        group = gmsh.model.addPhysicalGroup(1, tags)
        gmsh.model.setPhysicalName(1, group, edge_name)

    # gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)  # 设置最小网格大小
    # gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)  # 设置最大网格大小

    f = gmsh.model.mesh.field
    f.add("Distance",1)
    f.setNumbers(1,"CurvesList",[arc])
    f.setNumbers(1,"PointsList",[2,3])
    f.setNumber(1,"Sampling", 100)

    f.add("Threshold", 2)
    f.setNumber(2, "IField", 1)
    f.setNumber(2, "LcMin", 0.01)    # 接触边附近最小尺寸
    f.setNumber(2, "LcMax", 0.075)    # 远场最大尺寸
    f.setNumber(2, "DistMin", 0.05)
    f.setNumber(2, "DistMax", 0.055)

    f.setAsBackgroundMesh(2)  # 设置为背景网格

    gmsh.option.setNumber("Mesh.Algorithm", 5)  # 设置最小网格大小

    gmsh.model.mesh.setTransfiniteCurve(6, div_num_w+1)
    gmsh.model.mesh.setTransfiniteCurve(4, div_num_w+1)
    gmsh.model.mesh.setTransfiniteCurve(7, div_num_h+1)
    gmsh.model.mesh.setTransfiniteCurve(5, div_num_h+1)

    # 生成结构网格
    # gmsh.model.mesh.setTransfiniteSurface(1, "Alternate")  # 设置跨限算法
    gmsh.model.mesh.setTransfiniteSurface(2, "Alternate")  # 设置跨限算法
    # gmsh.model.mesh.setRecombine(2, 1)  # 重组为四边形网格
    gmsh.model.mesh.setRecombine(2, 2)  # 重组为四边形网格
    gmsh.model.mesh.generate(2)  # 生成2D网格

    # 保存文件
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(file_name)
    print(f"Gmsh OCC模型已成功写入文件: {file_name}")

    gmsh.fltk.run()

    gmsh.finalize()

    # 启动gmsh
        
def create_hertz_model_refine(file_name:str,refine_level:int=1):
    """
    创建一个Hertz接触模型的网格文件，使用更细的网格。
    """
    # 使用Gmsh OCC API创建Hertz模型
    gmsh.initialize()
    gmsh.model.add("Hertz_Model")
    f = gmsh.model.occ
    # 定义Hertz模型的几何

    #  geometry
    #  ┌───────────────6───────────────┐
    #  7                               5
    #  └───────────────4───────────────┘
    #
    #  
    #  * *
    #  *       *
    #  *            *
    #  *               1 
    #  *                 *  
    #  2                   *
    #  *                     *
    #  *                      *
    #  *                       *
    #  *   *   *   3   *   *   *
    
    radius = 1.0


    p_center = f.addPoint(0, 0, 0)    
    p_outer_y = f.addPoint(0, radius, 0)
    p_outer_x = f.addPoint(radius, 0, 0)
    
    arc = f.addCircleArc(p_outer_y, p_center, p_outer_x) # 创建上半圆弧
    line_y = f.addLine(p_center, p_outer_y)
    line_x = f.addLine(p_outer_x, p_center)  # 创建水平线
    
    cloop = f.addCurveLoop([line_y, arc, line_x])
    surf_circle = f.addPlaneSurface([cloop]) 

    # cut_circle = f.addCircle(0, radius, 0, 0.1)  # 添加一个半径为0.1的圆

    p_cut_1 = f.addPoint(0, radius - 0.05*radius, 0)
    p_cut_2 = f.addPoint(0.08*radius, radius - 0.05*radius, 0)
    p_cut_3 = f.addPoint(0.08*radius, radius, 0)

    p_cut_4 = f.addPoint(0, radius - 0.2*radius, 0)
    p_cut_5 = f.addPoint(0.2*radius, radius - 0.2*radius, 0)
    p_cut_6 = f.addPoint(0.26*radius, radius, 0)
    p_cut_7 = f.addPoint(0.5*radius, 0, 0)


    line_cut_1 = f.addLine(p_cut_1, p_cut_2)
    line_cut_2 = f.addLine(p_cut_2, p_cut_3)
    line_cut_3 = f.addLine(p_cut_2, p_cut_5)

    line_cut_4 = f.addLine(p_cut_4, p_cut_5)
    line_cut_5 = f.addLine(p_cut_5, p_cut_6)
    line_cut_6 = f.addLine(p_cut_5, p_cut_7)

    line_cuts_circle = [(1, l) for l in [line_cut_1, line_cut_2, line_cut_3, line_cut_4, line_cut_5, line_cut_6]]
    f.fragment([(2, surf_circle)], line_cuts_circle)  # 将圆切割

    # 定义一个长方形
    width = 1.5  # 长方形的宽度
    height = 0.1  # 长方形的高度
    y_offset = 1*height  # 长方形的y偏移量
    surf_rectangle = f.addRectangle(0, radius + y_offset, 0, width, height)  # 定义一个长方形

    p_cut_8 = f.addPoint(0.3*radius, radius + 1.1*y_offset + height, 0)
    p_cut_9 = f.addPoint(0.3*radius, radius + 0.9*y_offset, 0)
    line_cut7= f.addLine(p_cut_8, p_cut_9)
    f.fragment([(2, surf_rectangle)], [(1,line_cut7)])  # 将圆切割

    f.synchronize()

    # 定义物理组
    # 定义半圆的物理组
    body_group_specimen = gmsh.model.addPhysicalGroup(2, [1,2,3,4,5])
    gmsh.model.setPhysicalName(2, body_group_specimen, "Specimen_Body")

    # 定义长方形的物理组
    body_group_indenter = gmsh.model.addPhysicalGroup(2, [6,7])
    gmsh.model.setPhysicalName(2, body_group_indenter, "Indenter_Body")

    bc_edge_group_dict = {
        "Indenter_Top_Edge": [23, 26],
        "Indenter_Left_Edge": [22],
        "Indenter_Bottom_Edge": [25, 28],
        "Specimen_Top_Edge": [17, 19, 14],
        "Specimen_Left_Edge": [18, 12, 11],
        "Specimen_Bottom_Edge": [10, 13]
    }
    for edge_name, tags in bc_edge_group_dict.items():
        group = gmsh.model.addPhysicalGroup(1, tags)
        gmsh.model.setPhysicalName(1, group, edge_name)

    N = 5*refine_level  # 网格细化级别
    div_num_dict = {
        12: 8*N,
        6: 8*N,
        19: 8*N,
        11: 8*N,
        9: 8*N,
        14: 8*N,
        17: 3*N,
        4: 3*N,
        7: 3*N,
        10: 3*N,
        23: 4*N,
        25: 4*N,
        22: 2,        
        24: 2,
        27: 2,
    }

    # 遍历所有的曲面
    for surf_id in [1,2,3,4,5,6,7]:
        edges = gmsh.model.getBoundary([(2, surf_id)], combined=True, oriented=False)
        # 遍历所有的边
        for edge in edges:
            edge_tag = edge[1]
            # 设置每条边的跨限数
            if edge_tag in div_num_dict:
                gmsh.model.mesh.setTransfiniteCurve(edge_tag, div_num_dict[edge_tag]+1)
            else:
                gmsh.model.mesh.setTransfiniteCurve(edge_tag, 3*N+1)

        gmsh.model.mesh.setTransfiniteSurface(surf_id, "Alternate")  # 设置跨限算法
        gmsh.model.mesh.setRecombine(2, surf_id)  # 重组为四边形网格

    gmsh.model.mesh.generate(2)  # 生成2D网格
    gmsh.fltk.run()
    # 保存文件
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(file_name)
    print(f"Gmsh OCC模型已成功写入文件: {file_name}")

def create_hertz_model_structural_mesh(file_name:str,refine_level:int=1):
    """
    创建一个Hertz接触模型的网格文件，使用更细的网格。
    """
    # 使用Gmsh OCC API创建Hertz模型
    gmsh.initialize()
    gmsh.model.add("Hertz_Model")
    f = gmsh.model.occ
    # 定义Hertz模型的几何

    #  geometry
    #  ┌───────────────6───────────────┐
    #  7                               5
    #  └───────────────4───────────────┘
    #
    #  
    #  * *
    #  *       *
    #  *            *
    #  *               1 
    #  *                 *  
    #  2                   *
    #  *                     *
    #  *                      *
    #  *                       *
    #  *   *   *   3   *   *   *
    
    radius = 1.0


    p_center = f.addPoint(0, 0, 0)    
    p_outer_y = f.addPoint(0, radius, 0)
    p_outer_x = f.addPoint(radius, 0, 0)
    
    arc = f.addCircleArc(p_outer_y, p_center, p_outer_x) # 创建上半圆弧
    line_y = f.addLine(p_center, p_outer_y)
    line_x = f.addLine(p_outer_x, p_center)  # 创建水平线
    
    cloop = f.addCurveLoop([line_y, arc, line_x])
    surf_circle = f.addPlaneSurface([cloop]) 

    # cut_circle = f.addCircle(0, radius, 0, 0.1)  # 添加一个半径为0.1的圆

    # p_cut_1 = f.addPoint(0, radius - 0.05*radius, 0)
    # p_cut_2 = f.addPoint(0.08*radius, radius - 0.05*radius, 0)
    # p_cut_3 = f.addPoint(0.08*radius, radius, 0)

    p_cut_4 = f.addPoint(0, radius - 0.2*radius, 0)
    p_cut_5 = f.addPoint(0.2*radius, radius - 0.2*radius, 0)
    p_cut_6 = f.addPoint(0.26*radius, radius, 0)
    p_cut_7 = f.addPoint(0.5*radius, 0, 0)


    # line_cut_1 = f.addLine(p_cut_1, p_cut_2)
    # line_cut_2 = f.addLine(p_cut_2, p_cut_3)
    # line_cut_3 = f.addLine(p_cut_2, p_cut_5)

    line_cut_4 = f.addLine(p_cut_4, p_cut_5)
    line_cut_5 = f.addLine(p_cut_5, p_cut_6)
    line_cut_6 = f.addLine(p_cut_5, p_cut_7)

    line_cuts_circle = [(1, l) for l in [line_cut_4, line_cut_5, line_cut_6]]
    f.fragment([(2, surf_circle)], line_cuts_circle)  # 将圆切割

    # 定义一个长方形
    width = 1.2  # 长方形的宽度
    height = 0.1  # 长方形的高度
    y_offset = 1*height  # 长方形的y偏移量
    surf_rectangle = f.addRectangle(0, radius + y_offset, 0, width, height)  # 定义一个长方形

    # p_cut_8 = f.addPoint(0.3*radius, radius + 1.1*y_offset + height, 0)
    # p_cut_9 = f.addPoint(0.3*radius, radius + 0.9*y_offset, 0)
    # line_cut7= f.addLine(p_cut_8, p_cut_9)
    # f.fragment([(2, surf_rectangle)], [(1,line_cut7)])  # 将圆切割

    f.synchronize()

    # 定义物理组
    # 定义半圆的物理组
    circle_surf_tags = [1,2,3]
    body_group_specimen = gmsh.model.addPhysicalGroup(2, [1,2,3])
    gmsh.model.setPhysicalName(2, body_group_specimen, "Specimen_Body")

    # 定义长方形的物理组
    rect_surf_tags = [4]
    body_group_indenter = gmsh.model.addPhysicalGroup(2, [4])
    gmsh.model.setPhysicalName(2, body_group_indenter, "Indenter_Body")

    bc_edge_group_dict = {
        "Specimen_Top_Edge": [10, 13],
        "Specimen_Left_Edge": [11, 8],
        "Specimen_Bottom_Edge": [7, 12],        
        "Indenter_Top_Edge": [17],
        "Indenter_Left_Edge": [18],
        "Indenter_Bottom_Edge": [15]
    }
    for edge_name, tags in bc_edge_group_dict.items():
        group = gmsh.model.addPhysicalGroup(1, tags)
        gmsh.model.setPhysicalName(1, group, edge_name)

    N = 2*refine_level  # 网格细化级别

    edge_config_groups = [
        {"ids":[6,8,13],"name":"","div_num":10*N,"progression_factor":1.0},
        {"ids":[18,16],"name":"","div_num":3,"progression_factor":1.0},
        {"ids":[15,17],"name":"","div_num":8*N,"progression_factor":1.0}
        # {"ids" : [12,6,19],"name":"半圆_中_纵","div_num" : 8*N,"progression_factor":1.0},
        # {"ids" : [11,9,14],"name":"半圆_下_纵","div_num" : 8*N,"progression_factor":1.0},
        # {"ids" : [17,4,7,10],"name":"半圆_左","div_num" : 3*N,"progression_factor":1.0},
        # {"ids" : [18,16,15,13],"name":"半圆_右","div_num" : 3*N,"progression_factor":1.0},
        # {"ids" : [23,25],"name":"矩形_左_横","div_num" : 8*N,"progression_factor":1.0},
        # {"ids" : [26],"name":"矩形_右上_横","div_num" : 8*N,"progression_factor":-1.1},
        # {"ids" : [28],"name":"矩形_右下_横","div_num" : 8*N,"progression_factor":1.1},
        # {"ids" : [22,24],"name":"矩形_左_纵","div_num" : 4*N,"progression_factor":-1.3},
        # {"ids" : [27],"name":"矩形_右_纵","div_num" : 4*N,"progression_factor":1.3}
    ]

    # 遍历所有的曲面
    for surf_id in circle_surf_tags + rect_surf_tags:
        edges = gmsh.model.getBoundary([(2, surf_id)], combined=True, oriented=False)
        # 遍历所有的边
        for edge in edges:
            edge_tag = edge[1]
            # 设置每条边的跨限数
            for config in edge_config_groups:
                if edge_tag in config["ids"]:
                    div_num = int(config["div_num"])
                    progression_factor = config["progression_factor"]
                    gmsh.model.mesh.setTransfiniteCurve(edge_tag, div_num+1,"Progression", progression_factor)
                    break
            else:
                gmsh.model.mesh.setTransfiniteCurve(edge_tag, int(3*N)+1)

        gmsh.model.mesh.setTransfiniteSurface(surf_id, "Alternate")  # 设置跨限算法
        gmsh.model.mesh.setRecombine(2, surf_id)  # 重组为四边形网格


    gmsh.model.mesh.generate(2)  # 生成2D网格
    gmsh.fltk.run()
    # 保存文件
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(file_name)
    print(f"Gmsh OCC模型已成功写入文件: {file_name}")

def create_hertz_model_half_space(file_name:str,refine_level:int=1):
    """
    创建一个Hertz接触模型的网格文件，使用更细的网格。
    """
    # 使用Gmsh OCC API创建Hertz模型
    gmsh.initialize()
    gmsh.model.add("Hertz_Model")
    f = gmsh.model.occ
    # 定义Hertz模型的几何

    #  geometry
    #  ┌───────────────6───────────────┐
    #  7                               5
    #  └───────────────4───────────────┘
    #
    #  
    #  * *
    #  *       *
    #  *            *
    #  *               1 
    #  *                 *  
    #  2                   *
    #  *                     *
    #  *                      *
    #  *                       *
    #  *   *   *   3   *   *   *
    
    radius = 1.0


    p_center = f.addPoint(0, 0, 0)    
    p_outer_y = f.addPoint(0, radius, 0)
    p_outer_x = f.addPoint(radius, 0, 0)
    
    arc = f.addCircleArc(p_outer_y, p_center, p_outer_x) # 创建上半圆弧
    line_y = f.addLine(p_center, p_outer_y)
    line_x = f.addLine(p_outer_x, p_center)  # 创建水平线
    
    cloop = f.addCurveLoop([line_y, arc, line_x])
    surf_circle = f.addPlaneSurface([cloop]) 

    # cut_circle = f.addCircle(0, radius, 0, 0.1)  # 添加一个半径为0.1的圆

    p_cut_1 = f.addPoint(0, radius - 0.05*radius, 0)
    p_cut_2 = f.addPoint(0.08*radius, radius - 0.05*radius, 0)
    p_cut_3 = f.addPoint(0.08*radius, radius, 0)

    p_cut_4 = f.addPoint(0, radius - 0.2*radius, 0)
    p_cut_5 = f.addPoint(0.2*radius, radius - 0.2*radius, 0)
    p_cut_6 = f.addPoint(0.26*radius, radius, 0)
    p_cut_7 = f.addPoint(0.5*radius, 0, 0)


    line_cut_1 = f.addLine(p_cut_1, p_cut_2)
    line_cut_2 = f.addLine(p_cut_2, p_cut_3)
    line_cut_3 = f.addLine(p_cut_2, p_cut_5)

    line_cut_4 = f.addLine(p_cut_4, p_cut_5)
    line_cut_5 = f.addLine(p_cut_5, p_cut_6)
    line_cut_6 = f.addLine(p_cut_5, p_cut_7)

    line_cuts_circle = [(1, l) for l in [line_cut_1, line_cut_2, line_cut_3, line_cut_4, line_cut_5, line_cut_6]]
    f.fragment([(2, surf_circle)], line_cuts_circle)  # 将圆切割

    # 定义一个长方形
    width = 20  # 长方形的宽度
    height = 5  # 长方形的高度
    y_offset = 0.1  # 长方形的y偏移量
    surf_rectangle = f.addRectangle(0, radius + y_offset, 0, width, height)  # 定义一个长方形

    p_cut_8 = f.addPoint(0.3*radius, radius + 1.1*y_offset + height, 0)
    p_cut_9 = f.addPoint(0.3*radius, radius + 0.9*y_offset, 0)
    line_cut7= f.addLine(p_cut_8, p_cut_9)
    f.fragment([(2, surf_rectangle)], [(1,line_cut7)])  # 将圆切割

    f.synchronize()

    # 定义物理组
    # 定义半圆的物理组
    body_group_specimen = gmsh.model.addPhysicalGroup(2, [1,2,3,4,5])
    gmsh.model.setPhysicalName(2, body_group_specimen, "Specimen_Body")

    # 定义长方形的物理组
    body_group_indenter = gmsh.model.addPhysicalGroup(2, [6,7])
    gmsh.model.setPhysicalName(2, body_group_indenter, "Indenter_Body")

    bc_edge_group_dict = {
        "Indenter_Top_Edge": [23, 26],
        "Indenter_Left_Edge": [22],
        "Indenter_Bottom_Edge": [25, 28],
        "Specimen_Top_Edge": [17, 19, 14],
        "Specimen_Left_Edge": [18, 12, 11],
        "Specimen_Bottom_Edge": [10, 13]
    }
    for edge_name, tags in bc_edge_group_dict.items():
        group = gmsh.model.addPhysicalGroup(1, tags)
        gmsh.model.setPhysicalName(1, group, edge_name)

    N = 5*refine_level  # 网格细化级别

    edge_config_groups = [
        {"ids" : [12,6,19],"name":"半圆_中_纵","div_num" : 8*N,"progression_factor":1.0},
        {"ids" : [11,9,14],"name":"半圆_下_纵","div_num" : 8*N,"progression_factor":1.0},
        {"ids" : [17,4,7,10],"name":"半圆_左","div_num" : 3*N,"progression_factor":1.0},
        {"ids" : [18,16,15,13],"name":"半圆_右","div_num" : 3*N,"progression_factor":1.0},
        {"ids" : [23,25],"name":"矩形_左_横","div_num" : 8*N,"progression_factor":1.0},
        {"ids" : [26],"name":"矩形_右上_横","div_num" : 8*N,"progression_factor":-1.1},
        {"ids" : [28],"name":"矩形_右下_横","div_num" : 8*N,"progression_factor":1.1},
        {"ids" : [22,24],"name":"矩形_左_纵","div_num" : 4*N,"progression_factor":-1.3},
        {"ids" : [27],"name":"矩形_右_纵","div_num" : 4*N,"progression_factor":1.3}
    ]

    # 遍历所有的曲面
    for surf_id in [1,2,3,4,5,6,7]:
        edges = gmsh.model.getBoundary([(2, surf_id)], combined=True, oriented=False)
        # 遍历所有的边
        for edge in edges:
            edge_tag = edge[1]
            # 设置每条边的跨限数
            for config in edge_config_groups:
                if edge_tag in config["ids"]:
                    div_num = config["div_num"]
                    progression_factor = config["progression_factor"]
                    gmsh.model.mesh.setTransfiniteCurve(edge_tag, div_num+1,"Progression", progression_factor)
                    break
            else:
                gmsh.model.mesh.setTransfiniteCurve(edge_tag, 3*N+1)

        gmsh.model.mesh.setTransfiniteSurface(surf_id, "Alternate")  # 设置跨限算法
        gmsh.model.mesh.setRecombine(2, surf_id)  # 重组为四边形网格

    gmsh.model.mesh.generate(2)  # 生成2D网格
    gmsh.fltk.run()
    # 保存文件
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(file_name)
    print(f"Gmsh OCC模型已成功写入文件: {file_name}")

if __name__ == "__main__":

    # 调用函数创建网格
    # create_mesh_with_gmsh_occ("contact2d.msh")
    # create_hertz_model("hertz_contact.msh")
    # create_hertz_model_refine("hertz_contact_refine2.msh",refine_level=2)
    # create_hertz_model_half_space("hertz_contact_half_space.msh",refine_level=1)
    # create_hertz_model_contact_area_refine("hertz_contact_refine.msh")
    create_hertz_model_structural_mesh("hertz_contact_structural_ref_1.5.msh",1.5)    #bottom edge div num : 30*1