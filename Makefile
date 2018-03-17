CXX = g++
FLAGS = -std=c++11 -O3 -march=native
# FLAGS = -std=c++11 -g
INCLUDES = -I/usr/local/include/opencv -I/usr/local/include -I/usr/local/include/eigen3/
LIBS = -lm -L/usr/local/lib -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core

TARGET = main
SRCS = main.cc pencildrawer.cc

OBJS = $(patsubst %.cc,%.o,$(SRCS))

.cc.o:
	$(CXX) $(FLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJS)
		$(CXX) $(FLAGS) $(FLAG) -o $@ $(OBJS) $(LIBS)

clean:
	rm $(TARGET) $(OBJS)

run:
	./$(TARGET)
