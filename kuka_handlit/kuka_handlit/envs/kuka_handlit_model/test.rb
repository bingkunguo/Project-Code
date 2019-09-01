require 'yaml'


class Kuka
    attr_accessor :lbr_iiwa_link_0,:lbr_iiwa_link_1,:lbr_iiwa_link_2,:lbr_iiwa_link_3,:lbr_iiwa_link_4,:lbr_iiwa_link_5,:lbr_iiwa_link_6,:lbr_iiwa_link_7,:mesh_v,:mesh_c,:model
    def initialize(sim_env)
      @lbr_iiwa_link_0={
        # link 
        "link_name"     =>"lbr_iiwa_link_0",
        'pose' =>[0,0,0,
                0,0,0],
        #inertia
        'pose_i'=>[-0.1,0,0.07,
                   0,0,0],
        'mass'=>0.0,
        'ixx'=> 0.05 ,'ixy'=> 0     ,'ixz'=> 0 ,
        'iyy'=> 0.06 ,'iyz'=> 0     ,'izz'=>0.03,
        #mesh
        'scale'=>[1,1,1],
        #material
        'ambient'  => [1,0,0,1],
        'diffuse'  => [0.2,0.2,0.2,1.0],
        'specular' => [0.4,0.4,0.4,1],
        'emissive' => [0,0,0,0]
       
      }
      @lbr_iiwa_link_1={
        #link
        "link_name"       =>"lbr_iiwa_link_1",
        'pose' =>[0,0,0.1575,
                0,0,0],
        #inertia
        'pose_i'=>[0,0.03 ,0.12,
                   0,0,0],
        'mass'=>4.0,
        'ixx'=> 0.1 ,'ixy'=> 0     ,'ixz'=> 0 ,
        'iyy'=> 0.09 ,'iyz'=> 0     ,'izz'=>0.02,
        #mesh
        'scale'=>[1,1,1],
        #material
        'ambient'  => [1,0,0,1],
        'diffuse'  => [0.5 ,0.7 ,1.0 ,1.0],
        'specular' => [0.5 ,0.5 ,0.5, 1],
        'emissive' => [0,0,0,0],
        # joint 
        "p_name"     =>"lbr_iiwa_link_0",
        "j_name"     =>"J0",
        "j_type"     =>"revolute",
        "j_axis"     => [0,0,1],
        "j_limit"    => [-2.96706,2.96706], #[lower,upper]
        "j_effort"   => 300,
        "j_velocity" => 10,
        "j_damping"  => 0.5, 
        "friction"   => 0,
        "spring_reference" => 0,
        "spring_stiffness" => 0
      }
      @lbr_iiwa_link_2={
        #link
        "link_name"       =>"lbr_iiwa_link_2",
        'pose' =>[0,0,0.36,
                 1.5708,0,-3.14159],
        #inertia
        'pose_i'=>[0.0003,0.059,0.042,
                   0,0,0],
        'mass'=>4.0,
        'ixx'=> 0.05  ,'ixy'=> 0     ,'ixz'=> 0 ,
        'iyy'=> 0.018 ,'iyz'=> 0     ,'izz'=>044,
        #mesh
        'scale'=>[1,1,1],
        #material
        'ambient'  => [1,0,0,1],
        'diffuse'  => [0.5 ,0.7 ,1.0 ,1.0],
        'specular' => [0.5 ,0.5 ,0.5 ,1],
        'emissive' => [0,0,0,0],
        # joint 
        "p_name"     =>"lbr_iiwa_link_1",
        "j_name"     =>"J1",
        "j_type"     =>"revolute",
        "j_axis"     => [0,0,1],
        "j_limit"    => [-2.0944,2.0944], #[lower,upper]
        "j_effort"   => 300,
        "j_velocity" => 10,
        "j_damping"  => 0.5, 
        "friction"   => 0,
        "spring_reference" => 0,
        "spring_stiffness" => 0
      }
      @lbr_iiwa_link_3={
        #link
        "link_name"       =>"lbr_iiwa_link_3",
        'pose' =>[0,0, 0.5645,
                0,0,0],
        #inertia
        'pose_i'=>[0,0.03 ,0.13,
                   0,0,0],
        'mass'=>3.0,
        'ixx'=> 0.08 ,'ixy'=> 0     ,'ixz'=> 0 ,
        'iyy'=> 0.075 ,'iyz'=> 0     ,'izz'=>0.01,
        #mesh
        'scale'=>[1,1,1],
        #material
        'ambient'  => [1,0,0,1],
        'diffuse'  => [1.0 ,0.423529411765 ,0.0392156862745 ,1.0],
        'specular' => [0.5 ,0.5 ,0.5 ,1],
        'emissive' => [0,0,0,0],
        # joint 
        "p_name"     =>"lbr_iiwa_link_2",
        "j_name"     =>"J2",
        "j_type"     =>"revolute",
        "j_axis"     => [0,0,1],
        "j_limit"    => [-2.96706,2.96706], #[lower,upper]
        "j_effort"   => 300,
        "j_velocity" => 10,
        "j_damping"  => 0.5, 
        "friction"   => 0,
        "spring_reference" => 0,
        "spring_stiffness" => 0
      }
      @lbr_iiwa_link_4={
        #link
        "link_name"       =>"lbr_iiwa_link_4",
        'pose' =>[0,0,0.78,
                  1.5708,0,0],
        #inertia
        'pose_i'=>[0,0.067,0,
                   0.034,0,0],
        'mass'=>2.7,
        'ixx'=> 0.03 ,'ixy'=> 0     ,'ixz'=> 0 ,
        'iyy'=> 0.01 ,'iyz'=> 0     ,'izz'=>0.029,
        #mesh
        'scale'=>[1,1,1],
        #material
        'ambient'  => [1,0,0,1],
        'diffuse'  => [0.5,0.7,1.0,1.0],
        'specular' => [0.5,0.5,0.5,1],
        'emissive' => [0,0,0,0],
        # joint 
        "p_name"     =>"lbr_iiwa_link_3",
        "j_name"     =>"J3",
        "j_type"     =>"revolute",
        "j_axis"     => [0,0,1],
        "j_limit"    => [-2.0944,2.0944], #[lower,upper]
        "j_effort"   => 300,
        "j_velocity" => 10,
        "j_damping"  => 0.5, 
        "friction"   => 0,
        "spring_reference" => 0,
        "spring_stiffness" => 0
      }
      @lbr_iiwa_link_5={
        #link
        "link_name"       =>"lbr_iiwa_link_5",
        'pose' =>[0,0,0.9645,
                0,0,-3.14159],
        #inertia
        'pose_i'=>[0.0001,0.021,0.076,
                   0,0,0],
        'mass'=>1.7,
        'ixx'=> 0.02 ,'ixy'=> 0     ,'ixz'=> 0 ,
        'iyy'=> 0.018 ,'iyz'=> 0     ,'izz'=>0.005,
        #mesh
        'scale'=>[1,1,1],
        #material
        'ambient'  => [1,0,0,1],
        'diffuse'  => [0.5,0.7,1.0,1.0],
        'specular' => [0.5,0.5,0.5,1],
        'emissive' => [0,0,0,0],
        # joint 
        "p_name"     =>"lbr_iiwa_link_4",
        "j_name"     =>"J4",
        "j_type"     =>"revolute",
        "j_axis"     => [0,0,1],
        "j_limit"    => [-2.96706,2.96706], #[lower,upper]
        "j_effort"   => 300,
        "j_velocity" => 10,
        "j_damping"  => 0.5, 
        "friction"   => 0,
        "spring_reference" => 0,
        "spring_stiffness" => 0
      }
      @lbr_iiwa_link_6={
        #link
        "link_name"       =>"lbr_iiwa_link_6",
        'pose' =>[0,0,1.18,
                  1.5708,0,-3.14159],
        #inertia
        'pose_i'=>[0,0.0006,0.0004,
                   0,0,0],
        'mass'=>1.8,
        'ixx'=> 0.005 ,'ixy'=> 0     ,'ixz'=> 0 ,
        'iyy'=> 0.0036 ,'iyz'=> 0     ,'izz'=>0.0047,
        #mesh
        'scale'=>[1,1,1],
        #material
        'ambient'  => [1,0,0,1],
        'diffuse'  => [1.0,0.423529411765,0.0392156862745,1.0],
        'specular' => [0.5,0.5,0.5,1],
        'emissive' => [0,0,0,0],
        # joint 
        "p_name"     =>"lbr_iiwa_link_5",
        "j_name"     =>"J5",
        "j_type"     =>"revolute",
        "j_axis"     => [0,0,1],
        "j_limit"    => [-2.0944,2.0944], #[lower,upper]
        "j_effort"   => 300,
        "j_velocity" => 10,
        "j_damping"  => 0.5, 
        "friction"   => 0,
        "spring_reference" => 0,
        "spring_stiffness" => 0
      }
      @lbr_iiwa_link_7={
        #link
        "link_name"       =>"lbr_iiwa_link_7",
        'pose' =>[0,0,1.261,
                0,0,0],
        #inertia
        'pose_i'=>[0,0,0.02,
                   0,0,0],
        'mass'=>0.3,
        'ixx'=> 0.001 ,'ixy'=> 0     ,'ixz'=> 0 ,
        'iyy'=> 0.001 ,'iyz'=> 0     ,'izz'=>0.001,
        #mesh
        'scale'=>[1,1,1],
        #material
        'ambient'  => [1,0,0,1],
        'diffuse'  => [0.6,0.6,0.6,1],
        'specular' => [0.5,0.5,0.5,1],
        'emissive' => [0,0,0,0],
        # joint 
        "p_name"     =>"lbr_iiwa_link_6",
        "j_name"     =>"J6",
        "j_type"     =>"revolute",
        "j_axis"     => [0,0,1],
        "j_limit"    => [-3.05433,3.05433], #[lower,upper]
        "j_effort"   => 300,
        "j_velocity" => 10,
        "j_damping"  => 0.5, 
        "friction"   => 0,
        "spring_reference" => 0,
        "spring_stiffness" => 0
      }
      @mesh_c ={}
      @mesh_v ={}
      path_c = {"lbr_iiwa_link_0"=>"meshes/link_0.stl",
                "lbr_iiwa_link_1"=>"meshes/link_1.stl",
                "lbr_iiwa_link_2"=>"meshes/link_2.stl",
                "lbr_iiwa_link_3"=>"meshes/link_3.stl",
                "lbr_iiwa_link_4"=>"meshes/link_4.stl",
                "lbr_iiwa_link_5"=>"meshes/link_5.stl",
                "lbr_iiwa_link_6"=>"meshes/link_6.stl",
                "lbr_iiwa_link_7"=>"meshes/link_7.stl"
               }
      path_v = {"lbr_iiwa_link_0"=>"meshes/link_0.stl",
                "lbr_iiwa_link_1"=>"meshes/link_1.stl",
                "lbr_iiwa_link_2"=>"meshes/link_2.stl",
                "lbr_iiwa_link_3"=>"meshes/link_3.stl",
                "lbr_iiwa_link_4"=>"meshes/link_4.stl",
                "lbr_iiwa_link_5"=>"meshes/link_5.stl",
                "lbr_iiwa_link_6"=>"meshes/link_6.stl",
                "lbr_iiwa_link_7"=>"meshes/link_7.stl"
              }
      @model= {"KUKA"=>{"Links"=>{}}}
            
      def generate_model()
        kuka_link_list = [  @lbr_iiwa_link_0,@lbr_iiwa_link_1, @lbr_iiwa_link_2, @lbr_iiwa_link_3,
                            @lbr_iiwa_link_4, @lbr_iiwa_link_5, @lbr_iiwa_link_6, @lbr_iiwa_link_7]
        i=0
        loop do
            @model["KUKA"]["Links"]["lbr_iiwa_link_"+i.to_s] ={ 
                                                "pose"      => kuka_link_list[i]["pose"],
                                                "mass"      =>kuka_link_list[i]["mass"],
                                                "inertial"  => {
                                                    "pose_i" =>kuka_link_list[i]["pose_i"],
                                                    "ixx"    =>kuka_link_list[i]["ixx"],
                                                    "ixy"    =>kuka_link_list[i]["ixy"],
                                                    "ixz"    =>kuka_link_list[i]["ixz"],
                                                    "iyy"    =>kuka_link_list[i]["iyy"],
                                                    "iyz"    =>kuka_link_list[i]["iyz"],
                                                    "izz"    =>kuka_link_list[i]["izz"],
                                                    
                                                },
                                               # "color"     =>{
                                               #     'ambient'  => kuka_link_list[i]["ambient"] ,                                              'diffuse'  => .6,1],
                                               #     'specular' => kuka_link_list[i]["specular"],
                                               #     'emissive' => kuka_link_list[i]["emissive"],
                                               #     'diffuse' => kuka_link_list[i]["diffuse"] 
                                               # },
                                                "joint"     =>{
                                                    "p_name"    =>kuka_link_list[i]["p_name"],
                                                    "j_name"    =>kuka_link_list[i]["j_name"],
                                                    "j_type"    =>kuka_link_list[i]["j_type"],
                                                    "j_axis"    =>kuka_link_list[i]["j_axis"],
                                                    "j_limit"    =>kuka_link_list[i]["j_limit"],
                                                    "j_effort"    =>kuka_link_list[i]["j_effort"],
                                                    "j_velocity"    =>kuka_link_list[i]["j_velocity"],
                                                    "j_damping"    =>kuka_link_list[i]["j_damping"],
                                                    "friction"    =>kuka_link_list[i]["friction"]
                                                    
                                                }
                                           }
           
            if i ==7
                break
            end
            i +=1
        end
    
      end
      generate_model()
      if sim_env == "gazebo"
        #model://+default path
        path_c.each do |key,value|
          @mesh_c.merge!({key => "model://kuka_handlit/"+value})
        end
        path_v.each do |key,value|
          @mesh_v.merge!({key => "model://kuka_handlit/"+value})
        end

      elsif sim_env == "pybullet"
        #deafult path
        @mesh_v = path_v
        @mesh_c = path_c
      else
        puts "well life sucks!not everything is supported.A'nt life a bitch? "
      end


    end
end

class Base
    #constants
    M_PI = 3.1415926535897931
    attr_accessor :forearm,:palm,:mesh_c,:mesh_v,:model
    def initialize(sim_env,kuka_EE)
        @forearm ={  # link
                    'link_name'=>"forearm",
                    'pose'=> [0,0,0+kuka_EE["pose"][2],
                                   0,0,M_PI/2],
                    # inetia 
                    'pose_i'=> [0,0,0.09,
                                     0,0,0],                   
                    'mass'=> 1.8,
                    'ixx'=> 0.108 ,'ixy'=> 0     ,'ixz'=> 0 ,
                    'iyy'=> 0.108 ,'iyz'=> 0     ,'izz'=> 0.054 ,
                     #joint
                     "p_name" =>"lbr_iiwa_link_7",
                     "j_name" =>"kuka_hand",
                     "j_type" =>'fixed',
                    #color
                    "ambient"=>  [0,0,0,1],
                    "diffuse"=>  [0,0,0,1],
                    "specular"=> [0,0,0,1],
                    "emissive"=> [0,0,0,0]

        }

        @palm =  {  # link
                'link_name'=>"palm",
                'pose'=> [0+forearm['pose'][0],
                         -0.015+forearm['pose'][1],
                          0.149+forearm['pose'][2],
                         -40*M_PI/180,0,0],
                # inetia 
                'pose_i'=> [0,0,0.035,
                                0,0,0],                   
                'mass'=> 0.3,
                'ixx'=> 0.0003581 ,'ixy'=> 0     ,'ixz'=> 0 ,
                'iyy'=> 0.0005287 ,'iyz'=> 0     ,'izz'=> 0.000191 ,
                #joint
                "p_name" =>"forearm",
                "j_name" =>"palm_forearm",
                "j_type" =>'fixed',
                #color
                "ambient"=>  [0,0,0,1],
                "diffuse"=>  [0,0,0,1],
                "specular"=> [0,0,0,1],
                "emissive"=> [0,0,0,0]
        }
        @model={"BASE"=>{"Links"=>{}}}
        def loadYamlAndChange()
            file = YAML.load_file("config.yml")
            #print file 
         
                
            #print finger
            links = file["BASE"]["Links"]
            links.each do |linkName,link|

                color = link["color"]
                mass  = link["mass"]
                if !(linkName == "forearm")
                    joint = link["joint"]
                end
                

                if linkName == "forearm"
                    #visual
                    @forearm["ambient"] = color["ambient"]
                    @forearm["diffuse"] = color["diffuse"]
                    @forearm["specular"] = color["specular"]
                    @forearm["emissive"] = color["emissive"]
                    #physical prop
                    @forearm["mass"] = mass


                elsif linkName== "palm"
                    #visual
                    @palm["ambient"] = color["ambient"]
                    @palm["diffuse"] = color["diffuse"]
                    @palm["specular"] = color["specular"]
                    @palm["emissive"] = color["emissive"]
                    #physical prop
                    @palm["mass"] = mass


                end
            end
              
            
        end
        loadYamlAndChange()
        def generate_model()
            link_list = [ @forearm, @palm]
            i=0
            loop do
          
                @model["BASE"]["Links"][link_list[i]["link_name"]]={ 
                                                            "pose"      => link_list[i]["pose"],
                                                            "mass"      =>link_list[i]["mass"],
                                                            "inertial"  => {
                                                                "pose_i" =>link_list[i]["pose_i"],
                                                                "ixx"    =>link_list[i]["ixx"],
                                                                "ixy"    =>link_list[i]["ixy"],
                                                                "ixz"    =>link_list[i]["ixz"],
                                                                "iyy"    =>link_list[i]["iyy"],
                                                                "iyz"    =>link_list[i]["iyz"],
                                                                "izz"    =>link_list[i]["izz"],

                                                            },
                                                           # "color"     =>{
                                                           #     'ambient'  => link_list[i]["ambient"] ,                                              'diffuse'  => .6,1],
                                                           #     'specular' => link_list[i]["specular"],
                                                           #     'emissive' => link_list[i]["emissive"],
                                                           #     'diffuse' => link_list[i]["diffuse"] 
                                                           # },
                                                            "joint"     =>{
                                                                "p_name"        =>link_list[i]["p_name"],
                                                                "j_name"        =>link_list[i]["j_name"],
                                                                "j_type"        =>link_list[i]["j_type"],
                                                                "j_axis"        =>link_list[i]["j_axis"],
                                                                "j_limit"       =>link_list[i]["j_limit"],
                                                                "j_effort"      =>link_list[i]["j_effort"],
                                                                "j_velocity"    =>link_list[i]["j_velocity"],
                                                                "j_damping"     =>link_list[i]["j_damping"],
                                                                "friction"      =>link_list[i]["friction"]

                                                            }
                }
                if i ==1
                 break
                end
                i =+1
           end
           #print @model
        end
        generate_model()
        path_v ={
            "palm"      => "meshes/palm.dae",
            "forearm"   =>"meshes/forearm_lite.dae",
    
        }
        path_c ={
            "palm"      => "meshes/palm.dae",
            "forearm"   =>"meshes/forearm_lite.dae",
              
        }
        @mesh_c={}
        @mesh_v={}
        
        if sim_env == "gazebo"
            #model://+default path
            path_c.each do |key,value|
              @mesh_c.merge!({key => "model://shadow_handlit/"+value})
            end
            path_v.each do |key,value|
              @mesh_v.merge!({key => "model://shadow_handlit/"+value})
            end
        
        elsif sim_env == "pybullet"
            #deafult path
            @mesh_v = path_v
            @mesh_c = path_c
        else
            puts "Fingers:   well life sucks!not everything is supported.A'nt life a bitch? "
        end 
    end
end

class Finger
    #constants
    M_PI = 3.1415926535897931
    attr_accessor :knuckle,:proximal,:middle,:distal,:palm,:fingername,:mesh_c,:mesh_v,:model
    def initialize(sim_env,palm,biase,fingername)
        @fingername = fingername
        @palm = palm
        @knuckle ={ # link
                    'link_name'=>"knuckle_"+fingername,
                    'pose'=> [-0.033+palm['pose'][0]+biase[0],
                            0.0611+palm['pose'][1]+biase[1],
                            0.07277+palm['pose'][2]+biase[2],
                            0,0,0],
                    # inetia 
                    'pose_i'=> [0,0,0,
                                0,0,0],                   
                    'mass'=> 0.008,
                    'ixx'=> 0.00000026 ,'ixy'=> 0     ,'ixz'=> 0 ,
                    'iyy'=> 0.00000026 ,'iyz'=> 0     ,'izz'=>0.00000032,
                    # joint 
                    "p_name"       =>"palm",
                    "j_name"     =>"J4_"+fingername,
                    "j_type"     =>"revolute",
                    "j_axis"     => [0,-1,0],
                    "j_limit"    => [-20/180*M_PI,20/180*M_PI], #[lower,upper]
                    "j_effort"   => 2.0,
                    "j_velocity" => 2.0,
                    "j_damping"  => 0.1 ,
                    #color
                    "ambient"=>  [0,0,0,1],
                    "diffuse"=>  [0,0,0,1],
                    "specular"=> [0,0,0,1],
                    "emissive"=> [0,0,0,0]
        }
        @proximal = {  # link
                      'link_name'=>"proximal_"+fingername,
                      'pose'=> [0+knuckle['pose'][0],
                              0+knuckle['pose'][1],
                              0+knuckle['pose'][2],
                              0,0,0],
                      # inetia 
                      'pose_i'=> [0,0,0.0225,
                                      0,0,0],                   
                      'mass'=> 0.030,
                      'ixx'=> 0.0000098 ,'ixy'=> 0     ,'ixz'=> 0 ,
                      'iyy'=> 0.00001 ,'iyz'=> 0     ,'izz'=>0.0000018 ,
                       # joint 
                       "p_name"     =>"knuckle_"+fingername,
                       "j_name"     =>"J3_"+fingername,
                       "j_type"     =>"revolute",
                       "j_axis"     => [1,0,0],
                       "j_limit"    => [0,M_PI/2], #[lower,upper]
                       "j_effort"   => 2.0,
                       "j_velocity" => 2.0,
                       "j_damping"  => 0.1 ,
                        #color
                        "ambient"=>  [0,0,0,1],
                        "diffuse"=>  [0,0,0,1],
                        "specular"=> [0,0,0,1],
                        "emissive"=> [0,0,0,0]
                       
        } 
        @middle =   {  # link
                    'link_name'=>"middle_"+fingername,
                    'pose'=> [0+proximal['pose'][0],
                            0+proximal['pose'][1],
                            0.045+proximal['pose'][2],
                            0,0,0],
                    # inetia 
                    'pose_i'=> [0,0,0.0125,
                                    0,0,0],                   
                    'mass'=> 0.017,
                    'ixx'=>0.0000026 ,'ixy'=> 0     ,'ixz'=> 0 ,
                    'iyy'=>0.0000027 ,'iyz'=> 0     ,'izz'=>0.00000087,
                    # joint 
                    "p_name"     =>"proximal_"+fingername,
                    "j_name"     =>"J2_"+fingername,
                    "j_type"     =>"revolute",
                    "j_axis"     => [1,0,0],
                    "j_limit"    => [0,M_PI/2], #[lower,upper]
                    "j_effort"   => 2.0,
                    "j_velocity" => 2.0,
                    "j_damping"  => 0.1 ,
                    #color
                    "ambient"=>  [0,0,0,1],
                    "diffuse"=>  [0,0,0,1],
                    "specular"=> [0,0,0,1],
                    "emissive"=> [0,0,0,0]
        }
        @distal =   {  # link
                        'link_name'=>"distal_"+fingername,
                        'pose'=> [0+middle['pose'][0],
                                  0+middle['pose'][1],
                                  0.025+middle['pose'][2],
                                  0,0,0],
                        # inetia 
                        'pose_i'=> [0,0,0.012,
                                        0,0,0],                   
                        'mass'=> 0.012,
                        'ixx'=>0.00000094 ,'ixy'=> 0     ,'ixz'=> 0 ,
                        'iyy'=>0.0000011 ,'iyz'=> 0     ,'izz'=>0.00000053,
                        # joint 
                        "p_name"     =>"middle_"+fingername,
                        "j_name"     =>"J1_"+fingername,
                        "j_type"     =>"revolute",
                        "j_axis"     => [1,0,0],
                        "j_limit"    => [0,M_PI/2], #[lower,upper]
                        "j_effort"   => 2.0,
                        "j_velocity" => 2.0,
                        "j_damping"  => 0.1 ,
                        #color
                        "ambient"=>  [0,0,0,1],
                        "diffuse"=>  [0,0,0,1],
                        "specular"=> [0,0,0,1],
                        "emissive"=> [0,0,0,0]
        }
        @model={@fingername=>{"Links"=>{}}}

        def loadYamlAndChange()
            file = YAML.load_file("config.yml")
            #print file 
         
                
            #print finger
            links = file[ @fingername]["Links"]
            links.each do |linkName,link|

                color = link["color"]
                mass  = link["mass"] 
                joint = link["joint"]

                if linkName == "knuckle"
                    #visual
                    @knuckle["ambient"] = color["ambient"]
                    @knuckle["diffuse"] = color["diffuse"]
                    @knuckle["specular"] = color["specular"]
                    @knuckle["emissive"] = color["emissive"]
                    #physical prop
                    @knuckle["mass"] = mass
                    @knuckle["j_damping"] = joint["damping"]
                    @knuckle["j_friction"] = joint["friction"]
                elsif linkName== "proximal"
                    #visual
                    @proximal["ambient"] = color["ambient"]
                    @proximal["diffuse"] = color["diffuse"]
                    @proximal["specular"] = color["specular"]
                    @proximal["emissive"] = color["emissive"]
                    #physical prop
                    @proximal["mass"] = mass
                    @proximal["j_damping"] = joint["damping"]
                    @proximal["j_friction"] = joint["friction"]
                elsif linkName== "middle"
                    #visual
                    @middle["ambient"] = color["ambient"]
                    @middle["diffuse"] = color["diffuse"]
                    @middle["specular"] = color["specular"]
                    @middle["emissive"] = color["emissive"]
                    #physical prop
                    @middle["mass"] = mass
                    @middle["j_damping"] = joint["damping"]
                    @middle["j_friction"] = joint["friction"]
                elsif linkName== "distal"
                    #visual
                    @distal["ambient"] = color["ambient"]
                    @distal["diffuse"] = color["diffuse"]
                    @distal["specular"] = color["specular"]
                    @distal["emissive"] = color["emissive"]
                    #physical prop
                    @distal["mass"] = mass
                    @distal["j_damping"] = joint["damping"]
                    @distal["j_friction"] = joint["friction"]
                end
            end
              
            
        end
        loadYamlAndChange()
        def generate_model()
            link_list = [ @knuckle,@proximal,@middle,@distal]
            i= 0
         
            loop do
            
                @model[@fingername]["Links"][link_list[i]["link_name"]]={ 
                                                            "pose"      => link_list[i]["pose"],
                                                            "mass"      =>link_list[i]["mass"],
                                                            "inertial"  => {
                                                                "pose_i" =>link_list[i]["pose_i"],
                                                                "ixx"    =>link_list[i]["ixx"],
                                                                "ixy"    =>link_list[i]["ixy"],
                                                                "ixz"    =>link_list[i]["ixz"],
                                                                "iyy"    =>link_list[i]["iyy"],
                                                                "iyz"    =>link_list[i]["iyz"],
                                                                "izz"    =>link_list[i]["izz"],

                                                            },
                                                           # "color"     =>{
                                                           #     'ambient'  => link_list[i]["ambient"] ,                                              'diffuse'  => .6,1],
                                                           #     'specular' => link_list[i]["specular"],
                                                           #     'emissive' => link_list[i]["emissive"],
                                                           #     'diffuse' => link_list[i]["diffuse"] 
                                                           # },
                                                            "joint"     =>{
                                                                "p_name"        =>link_list[i]["p_name"],
                                                                "j_name"        =>link_list[i]["j_name"],
                                                                "j_type"        =>link_list[i]["j_type"],
                                                                "j_axis"        =>link_list[i]["j_axis"],
                                                                "j_limit"       =>link_list[i]["j_limit"],
                                                                "j_effort"      =>link_list[i]["j_effort"],
                                                                "j_velocity"    =>link_list[i]["j_velocity"],
                                                                "j_damping"     =>link_list[i]["j_damping"],
                                                                "friction"      =>link_list[i]["friction"]

                                                            }
                }
                if i ==link_list.length-1
                 break
                end
                i +=1
               
            end
           #print @model
        end
        generate_model()
        path_c = {  "distal"   => "meshes/F1.dae",
                    "middle"   => "meshes/F2.dae",
                    "proximal" => "meshes/F3.dae",
                    "knuckle"  => "meshes/knuckle.dae",
                    "palm"     => "meshes/palm.dae",
                    "forearm"  => "meshes/forearm_lite.dae",
        }
        path_v = {  "distal"   => "meshes/F1.dae",
                    "middle"   => "meshes/F2.dae",
                    "proximal" => "meshes/F3.dae",
                    "knuckle"  => "meshes/knuckle.dae"
        }
        @mesh_c={}
        @mesh_v={}
        if sim_env == "gazebo"
          #model://+default path
          path_c.each do |key,value|
            @mesh_c.merge!({key => "model://shadow_hand/"+value})
          end
          path_v.each do |key,value|
            @mesh_v.merge!({key => "model://shadow_hand/"+value})
          end

        elsif sim_env == "pybullet"
          #deafult path
          @mesh_v = path_v
          @mesh_c = path_c
        else
          puts "well life sucks!not everything is supported.A'nt life a bitch? "
        end
            
    end
end

class Thumb
    attr_accessor :thbase,:thproximal,:thhub,:thmiddle,:thdistal,:palm,:mesh_c,:mesh_v,:model
    #constants
    M_PI = 3.1415926535897931
    def initialize(sim_env,palm)
        @palm = palm
        @thbase =  {# link
                    'link_name'=>"thbase",
                    'pose'=> [-0.034+palm['pose'][0],
                    0.0085+palm['pose'][1],
                    0.029+palm['pose'][2],
                            0,0,0],
                    # inetia 
                    'pose_i'=> [0,0,0,
                                0,0,0],                   
                    'mass'=> 0.010,
                    'ixx'=> 0.00000016 ,'ixy'=> 0     ,'ixz'=> 0 ,
                    'iyy'=> 0.00000016 ,'iyz'=> 0     ,'izz'=>0.00000016 ,
                    # joint 
                    "p_name"       =>"palm",
                    "j_name"     =>"THJ5",
                    "j_type"     =>"revolute",
                    "j_axis"     => [0,0,1],
                    "j_limit"    => [-60/180*M_PI,60/180*M_PI], #[lower,upper]
                    "j_effort"   => 0.2,
                    "j_velocity" => 4.0,
                    "j_damping"  => 0.2,
                    #visual
                    "ambient"=>  [0,0,0,1],
                    "diffuse"=>  [0,0,0,1],
                    "specular"=> [0,0,0,1],
                    "emissive"=> [0,0,0,0]
                        
    
        } 
        @thproximal ={# link
                    'link_name'=>"thproximal",
                    'pose'=> [0+thbase['pose'][0],
                            0+thbase['pose'][1],
                            0+thbase['pose'][2],
                            0,0,0],
                    # inetia 
                    'pose_i'=> [0,0,0.019,
                                0,0,0],                   
                    'mass'=> 0.040,
                    'ixx'=> 0.0000136 ,'ixy'=> 0     ,'ixz'=> 0 ,
                    'iyy'=> 0.0000136 ,'iyz'=> 0     ,'izz'=>0.00000313,
                    # joint 
                    "p_name"     =>"thbase",
                    "j_name"     =>"THJ4",
                    "j_type"     =>"revolute",
                    "j_axis"     => [1,0,0],
                    "j_limit"    => [0,70/180*M_PI], #[lower,upper]
                    "j_effort"   => 3.0,
                    "j_velocity" => 4.0,
                    "j_damping"  => 0.2,
                    #visual
                    "ambient"=>  [0,0,0,1],
                    "diffuse"=>  [0,0,0,1],
                    "specular"=> [0,0,0,1],
                    "emissive"=> [0,0,0,0]                    
        } 
        @thhub = {  # link
                    'link_name' =>"thhub",
                    'pose'=> [0+thproximal['pose'][0],
                            0+thproximal['pose'][1],
                            0.038+thproximal['pose'][2],
                            0,0,0],
                    # inetia 
                    'pose_i'=> [0,0,0,
                                0,0,0],                   
                    'mass'=> 0.005,
                    'ixx'=> 0.000001 ,'ixy'=> 0     ,'ixz'=> 0 ,
                    'iyy'=> 0.000001 ,'iyz'=> 0     ,'izz'=>0.0000003 ,
                    # joint 
                    "p_name"     =>"thproximal",
                    "j_name"     =>"THJ3",
                    "j_type"     =>"fixed",
                    "j_axis"     => [1,0,0]
                }
        @thmiddle = {# link
                    'link_name' =>"thmiddle",
                    'pose'=> [0+thhub['pose'][0],
                            0+thhub['pose'][1],
                            0.01+thhub['pose'][2],
                            0,0,0],
                    # inetia 
                    'pose_i'=> [0,0,016,
                                0,0,0],                   
                    'mass'=> 0.020,
                    'ixx'=> 0.0000051 ,'ixy'=> 0     ,'ixz'=> 0 ,
                    'iyy'=> 0.0000051 ,'iyz'=> 0     ,'izz'=>0.00000121 ,
                    # joint 
                    "p_name"     =>"thhub",
                    "j_name"     =>"THJ2",
                    "j_type"     =>"revolute",
                    "j_axis"     => [0,-1,0],
                    "j_limit"    => [-40/180*M_PI,40/180*M_PI], #[lower,upper]
                    "j_effort"   => 2.0,
                    "j_velocity" => 2.0,
                    "j_damping"  => 0.1,
                    #visual
                    "ambient"=>  [0,0,0,1],
                    "diffuse"=>  [0,0,0,1],
                    "specular"=> [0,0,0,1],
                    "emissive"=> [0,0,0,0]
        } 
        @thdistal = {# link
                   'link_name' =>"thdistal",
                   'pose'=> [0+thmiddle['pose'][0],
                           0+thmiddle['pose'][1],
                           0.032+thmiddle['pose'][2],
                           0,0,M_PI/2],
                   # inetia 
                   'pose_i'=> [0,0,01375,
                               0,0,0],                   
                   'mass'=> 0.016,
                   'ixx'=> 0.0000021 ,'ixy'=> 0     ,'ixz'=> 0 ,
                   'iyy'=> 0.0000022 ,'iyz'=> 0     ,'izz'=>0.000001 ,
                    # joint 
                    "p_name"     =>"thmiddle",
                    "j_name"     =>"THJ1",
                    "j_type"     =>"revolute",
                    "j_axis"     => [1,0,0],
                    "j_limit"    => [0,90.0*M_PI/180], #[lower,upper]
                    "j_effort"   => 1.0,
                    "j_velocity" => 4.0,
                    "j_damping"  => 0.2,
                    #visual
                    "ambient"=>  [0,0,0,1],
                    "diffuse"=>  [0,0,0,1],
                    "specular"=> [0,0,0,1],
                    "emissive"=> [0,0,0,0]
        }
        @model={"TH"=>{"Links"=>{}}}
        def loadYamlAndChange()
            file = YAML.load_file("config.yml")
            #print file 
         
                
            #print finger
            links = file["TH"]["Links"]
            links.each do |linkName,link|

                color = link["color"]
                mass  = link["mass"] 
                joint = link["joint"]

                if linkName == "thbase"
                    #visual
                    @thbase["ambient"] = color["ambient"]
                    @thbase["diffuse"] = color["diffuse"]
                    @thbase["specular"] = color["specular"]
                    @thbase["emissive"] = color["emissive"]
                    #physical prop
                    @thbase["mass"] = mass
                    @thbase["j_damping"] = joint["damping"]
                    @thbase["j_friction"] = joint["friction"]
                elsif linkName== "thproximal"
                    #visual
                    @thproximal["ambient"] = color["ambient"]
                    @thproximal["diffuse"] = color["diffuse"]
                    @thproximal["specular"] = color["specular"]
                    @thproximal["emissive"] = color["emissive"]
                    #physical prop
                    @thproximal["mass"] = mass
                    @thproximal["j_damping"] = joint["damping"]
                    @thproximal["j_friction"] = joint["friction"]
                elsif linkName== "thhub"
                    #physical prop
                    @thhub["mass"] = mass
                    @thhub["j_damping"] = joint["damping"]
                    @thhub["j_friction"] = joint["friction"]
                elsif linkName== "thmiddle"
                    #visual
                    @thmiddle["ambient"] = color["ambient"]
                    @thmiddle["diffuse"] = color["diffuse"]
                    @thmiddle["specular"] = color["specular"]
                    @thmiddle["emissive"] = color["emissive"]
                    #physical prop
                    @thmiddle["mass"] = mass
                    @thmiddle["j_damping"] = joint["damping"]
                    @thmiddle["j_friction"] = joint["friction"]
                elsif linkName== "thdistal"
                    #visual
                    @thdistal["ambient"] = color["ambient"]
                    @thdistal["diffuse"] = color["diffuse"]
                    @thdistal["specular"] = color["specular"]
                    @thdistal["emissive"] = color["emissive"]
                    #physical prop
                    @thdistal["mass"] = mass
                    @thdistal["j_damping"] = joint["damping"]
                    @thdistal["j_friction"] = joint["friction"]
                end
            end
              
            
        end
        loadYamlAndChange()
        def generate_model()
            link_list = [@thbase,@thproximal,@thhub,@thmiddle,@thdistal]
            i= 0
         
            loop do
            
                @model["TH"]["Links"][link_list[i]["link_name"]]={ 
                                                            "pose"      => link_list[i]["pose"],
                                                            "mass"      =>link_list[i]["mass"],
                                                            "inertial"  => {
                                                                "pose_i" =>link_list[i]["pose_i"],
                                                                "ixx"    =>link_list[i]["ixx"],
                                                                "ixy"    =>link_list[i]["ixy"],
                                                                "ixz"    =>link_list[i]["ixz"],
                                                                "iyy"    =>link_list[i]["iyy"],
                                                                "iyz"    =>link_list[i]["iyz"],
                                                                "izz"    =>link_list[i]["izz"],

                                                            },
                                                           # "color"     =>{
                                                           #     'ambient'  => link_list[i]["ambient"] ,                                              'diffuse'  => .6,1],
                                                           #     'specular' => link_list[i]["specular"],
                                                           #     'emissive' => link_list[i]["emissive"],
                                                           #     'diffuse' => link_list[i]["diffuse"] 
                                                           # },
                                                            "joint"     =>{
                                                                "p_name"        =>link_list[i]["p_name"],
                                                                "j_name"        =>link_list[i]["j_name"],
                                                                "j_type"        =>link_list[i]["j_type"],
                                                                "j_axis"        =>link_list[i]["j_axis"],
                                                                "j_limit"       =>link_list[i]["j_limit"],
                                                                "j_effort"      =>link_list[i]["j_effort"],
                                                                "j_velocity"    =>link_list[i]["j_velocity"],
                                                                "j_damping"     =>link_list[i]["j_damping"],
                                                                "friction"      =>link_list[i]["friction"]

                                                            }
                }
                if i ==link_list.length-1
                 break
                end
                i +=1
               
            end
           #print @model
        end
        generate_model()
        path_v = {
                'thproximal' => "meshes/TH3_z.dae",
                'thmiddle'   => "meshes/TH2_z.dae",
                'thdistal'   => "meshes/TH1_z.dae"
        }
        path_c = {
                'thproximal' => "meshes/TH3_z.dae",
                'thmiddle'   => "meshes/TH2_z.dae",
                'thdistal'   => "meshes/TH1_z.dae"
        }
        @mesh_c ={}
        @mesh_v ={}
        
        if sim_env == "gazebo"
          #model://+default path
          path_c.each do |key,value|
            @mesh_c.merge!({key => "model://shadow_hand/"+value})
          end
          path_v.each do |key,value|
            @mesh_v.merge!({key => "model://shadow_hand/"+value})
          end
  
        elsif sim_env == "pybullet"
          #deafult path
          @mesh_v = path_v
          @mesh_c = path_c
        else
          puts "well life sucks!not everything is supported.A'nt life a bitch? "
        end


            
    end
end


sim_env ="pybullet" #change to pybullet|gazebo 

kuka = Kuka.new("pybullet")

base = Base.new(sim_env,kuka.lbr_iiwa_link_7)

biase = [0,0,0]
ff = Finger.new(sim_env,base.palm,biase,"FF")
biase = [0.022,0,0.004]
mf =  Finger.new(sim_env,base.palm,biase,"MF")
biase = [0.044,0,0]
rf  = Finger.new(sim_env,base.palm,biase,"RF")

thumb = Thumb.new(sim_env,base.palm)

def save_model(kuka,base,ff,mf,rf,thumb)
    robot_model = {
        "Hand"=>{
            "BASE"=>base.model["BASE"],
            "FF"=>ff.model["FF"],
            "MF"=>mf.model["MF"],
            "RF"=>rf.model["RF"],
            "TH"=>thumb.model["TH"]
             },
        "KUKA"=>kuka.model["KUKA"]
    }
    File.open("model_info.yml", "w") do |out|
        YAML.dump( robot_model, out )
         #{ |file| file.write(robot_model.to_yaml) }
    end
end
save_model(kuka,base,ff,mf,rf,thumb)
#i="boo"
#model= {i=>{}}
#model[i]["ass"]= "assFace"
#
#print model