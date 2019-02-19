/*
This sketch is for ventilate the room in an ordert way.
*/
const int flexPin = A0; //Define analog input pin to measure
            //flex sensor position. 

int ventilate_time = 0;

void setup() {
  Serial.begin(9600); //Set serial baud rate to 9600 bps
  // put your setup code here, to run once:
}

void loop() {
  int flexPosition;  // Input value from the analog pin.
 // put your main code here, to run repeatedly:

// Read the position of the flex sensor (0 to 1023):
  flexPosition = analogRead(flexPin);
  
  Serial.print("sensor: ");
  Serial.println(flexPosition);
  delay(20);
}
