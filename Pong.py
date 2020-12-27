import pygame
import cv2
import random

class  MyBallClass(pygame.sprite.Sprite):
    def __init__(self,image_file,speed=[10,5],location=[0,0]):#定义Ball类
        pygame.sprite.Sprite.__init__(self)
        
        self.image=pygame.image.load(image_file)
        self.rect=self.image.get_rect()
        self.rect.left,self.rect.top=location
        self.speed=speed
    def move(self):#定义move()方法
        global score,score_surf,score_font
        self.rect=self.rect.move(self.speed)
        if self.rect.left<0 or self.rect.right>screen.get_width():
            self.speed[0]=-self.speed[0]
        if self.rect.top<=0:   #球碰到屏幕顶部
            self.speed[1]=-self.speed[1]
            
class MyPaddleClass(pygame.sprite.Sprite):#定义球拍类
    def __init__(self,location=[0,0]):
        pygame.sprite.Sprite.__init__(self)
        image_surface=pygame.surface.Surface([100,10])
        image_surface.fill([0,0,0])
        self.image=image_surface.convert()
        self.rect=self.image.get_rect()
        self.rect.left,self.rect.top=location
        
pygame.init()
screen = pygame.display.set_mode([640,480])
clock = pygame.time.Clock()
Ball_vertical_speed = 4
Ball_horizontal_speed = 8
myBall = MyBallClass(r"ball.bmp",[Ball_horizontal_speed,Ball_vertical_speed],[50,50]) #球的速度和出现位置
ballGroup = pygame.sprite.Group(myBall)
paddle = MyPaddleClass([320,470])
lives = 3
score = 0
score_one_round = 0
offset = 5     #板子速度
score_font = pygame.font.Font(None,50) #创建font对象 50
score_surf = score_font.render(str(score),1,(0,0,0))
score_pos = [10,10]

def game_init():
    global lives,score,score_surf
    screen.fill([255,255,255])    
    screen.blit(myBall.image,myBall.rect)
    screen.blit(paddle.image,paddle.rect)
    screen.blit(score_surf,score_pos)
    for i in range(lives):#画出右上角显示为三条命的球
        width=screen.get_width()
        screen.blit(myBall.image,[width-40*i,20])
    pygame.display.flip()
    pygame.image.save(screen,"game.jpg") #有改进空间
    state = cv2.imread("game.jpg",0)
    return state

def game_step(action):
    global lives,score,score_surf,score_one_round
    
    done_show = False
    reward = 0
    clock.tick(60)
    screen.fill([255,255,255])
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return -1,-2

    #keys_pressed = pygame.key.get_pressed()                          #change this to trans the mode between random(machine control) and keyboard control
    #if keys_pressed[pygame.K_d] and paddle.rect.centerx < 590:       #change this to trans the mode between random(machine control) and keyboard control
    if action == 0 and paddle.rect.centerx < 590:                     #change this to trans the mode between keyboard control and random(machine control)
        paddle.rect.centerx += offset
    #if keys_pressed[pygame.K_a] and paddle.rect.centerx > 50:        #change this to trans the mode between random(machine control) and keyboard
    elif action == 2 and paddle.rect.centerx > 50:                    #change this to trans the mode between keyboard control and random(machine control)
        paddle.rect.centerx -= offset

    if pygame.sprite.spritecollide(paddle,ballGroup,False):#检测球与球拍的碰撞
        score= score + 1
        score_one_round = score_one_round + 1 
        score_surf = score_font.render(str(score),1,(0,0,0))
        if screen.get_rect().bottom - myBall.rect.bottom >= 2:       
            myBall.speed[1] =- myBall.speed[1]                        
            reward = 5                                             
            if abs(myBall.rect.centerx - paddle.rect.centerx) <= 12: 
                reward = 10
            elif abs(myBall.rect.centerx - paddle.rect.centerx) <= 30: 
                reward = 8
        else:
            myBall.speed[0]=-myBall.speed[0]
            print("bad")
            reward = 1
            
    myBall.move()
    screen.blit(myBall.image,myBall.rect)
    screen.blit(paddle.image,paddle.rect)
    screen.blit(score_surf,score_pos)
    
    for i in range(lives):#画出右上角显示为三条命的球
        width = screen.get_width()
        screen.blit(myBall.image,[width-40*i,20])
    pygame.display.flip()
        
    if myBall.rect.bottom >= screen.get_rect().bottom:#如果球碰到底边就减一条命,是否需要重置球拍位置？
        score_one_round = 0
        lives = lives-1
        reward = -5
        if abs(myBall.rect.centerx - paddle.rect.centerx) <= 100:
            reward = 0 
        elif abs(myBall.rect.centerx - paddle.rect.centerx) <= 200:
            reward = -3 
        if lives == 0:#创和绘制最终的分数文本
            f = open("scores.txt","a")   
            f.write("%d \n" % score)
            reward += -5 
            lives = 3
            score = 0
            score_surf = score_font.render(str(score),1,(0,0,0))
            
            final_text2 = "Game Over"
            ft2_font = pygame.font.Font(None,70)
            ft2_surf = ft2_font.render(final_text2,1,(0,0,0))
            screen.blit(ft2_surf,[screen.get_width()/2-ft2_surf.get_width()/2,100])
            pygame.display.flip()
            pygame.time.delay(1000)
            final_text3 = "Restart"
            ft3_font = pygame.font.Font(None,70)
            ft3_surf = ft3_font.render(final_text3,1,(0,0,0))
            screen.blit(ft3_surf,[screen.get_width()/2-ft3_surf.get_width()/2,200])
            pygame.display.flip()
            pygame.time.delay(1000)

            myBall.rect.topleft=[50,50]
            myBall.speed = [Ball_horizontal_speed,Ball_vertical_speed]  #Ball speed
            paddle.rect.left,paddle.rect.top=[320,470]
            done_show = True
        else:
            final_text4 = "left %s lives"%lives
            ft4_font = pygame.font.Font(None,70)
            ft4_surf = ft4_font.render(final_text4,1,(0,0,0))
            screen.blit(ft4_surf,[screen.get_width()/2-ft4_surf.get_width()/2,100])
            pygame.display.flip()
            pygame.time.delay(2000)
            myBall.rect.topleft = [50,50]
            myBall.speed = [Ball_horizontal_speed,Ball_vertical_speed]  #Ball speed
            paddle.rect.left,paddle.rect.top = [320,470]
    reward_1 = (50 - abs(myBall.rect.centerx - paddle.rect.centerx))/50
    if reward_1 < -0.00005:
        reward_1 = -0.00005
    
    reward_all = reward/10.0 *0.7 +  reward_1*0.3

    pygame.image.save(screen,"game.jpg") 
    state = cv2.imread("game.jpg",0)
    return state, reward_all, score_one_round,score ,done_show

def game_stop():
    pygame.quit()  
    
# use following code to run game without AI

# game_init()
# for i in range(0,10000):
#     action = random.choice([0, 1, 2])
#     a,b,c,d,e = game_step(action)
#     if b == -2:
#         game_stop()
#         break

