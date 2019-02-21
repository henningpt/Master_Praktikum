/*
This sketch is for ventilate the room in an ordert way.
*/
const int flexPin = A0; //Define analog input pin to measure
            //flex sensor position. 

unsigned long ventilate_time_total = 5000;
unsigned long ventilate_time_closed = 10000;   
unsigned long time_open = 0;
unsigned long time_closed = 0;
unsigned long timer = 0;

int flexPosition;  // Input value from the analog pin.

void setup() {
  Serial.begin(9600); //Set serial baud rate to 9600 bps
  pinMode(13, OUTPUT);
  pinMode(12, OUTPUT);
  // put your setup code here, to run once:
}


void loop() {
 // put your main code here, to run repeatedly:
 time_closed = time_difference(timer  ,millis()); 
 Serial.print("Time: ");
 Serial.println(millis()); //prints time since programm started

// Read the position of the flex sensor (0 to 1023):
  flexPosition = analogRead(flexPin);
  if(ventilate_time_closed < time_closed ){ //Is it time to ventilate?
  ventilate_process();
  }     
    
  Serial.print("sensor: ");
  Serial.println(flexPosition);
  delay(20);
}

unsigned long time_difference(unsigned long Start, unsigned long End){
    unsigned long difference; 
    difference = End - Start;
    return difference; 
  }

void ventilate_process(){
    while(flexPosition < 850){  // Is the window open?
      digitalWrite(13, HIGH);   // Turn on the red LED till the window is opened
    flexPosition = analogRead(flexPin); // read the position of the window
    }
   
   digitalWrite(13, LOW);   // Turn off the red LED 
   time_closed = 0;
   timer = millis(); // set the time to the current time
   
   while(ventilate_time_total < time_open){  //Was the window long enough open?
          time_open = time_difference(timer, millis());   
   }
   while(flexPosition > 850){  // Is the window closed?
          digitalWrite(12, HIGH);   // Turn on the yellow LED till the window is closed
          flexPosition = analogRead(flexPin); // read the position of the window
   }           
   
   digitalWrite(12, LOW);    // Turn off the yellow LED   
   timer = millis();    // set the new timer
  }  
  
