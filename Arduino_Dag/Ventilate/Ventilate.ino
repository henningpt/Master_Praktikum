/*
This sketch is for ventilate the room in an ordert way.
*/
const int flexPin = A0; //Define analog input pin to measure
            //flex sensor position. 

unsigned long ventilate_time_total = 600000;
unsigned long ventilate_time_closed = 8160000;   
unsigned long time_open = 0;
unsigned long time_closed = 0;
unsigned long timer = 0;
const int RED_PIN = 9;
const int GREEN_PIN = 10;
const int BLUE_PIN = 11;
int flexPosition;  // Input value from the analog pin.
const int button_pin = 2;  // pushbutton 2 pin
const int sensorPin = 1; // Light sensor
int speakerPin = 6;     //the pin that buzzer is connected to
unsigned long LightLevel; 
int button_state;
int season = 0; 

void setup() {
  Serial.begin(9600); //Set serial baud rate to 9600 bps
  pinMode(13, OUTPUT); 
  pinMode(12, OUTPUT);
  pinMode(speakerPin, OUTPUT);    //set the output pin for the speaker
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT);
  pinMode(button_pin, INPUT);

  // put your setup code here, to run once:
}


void loop() {
// determine the light level of the room 
LightLevel = analogRead(sensorPin);
Serial.print("Lichtintensität: ");
Serial.print(LightLevel);
//Is it night?
while(LightLevel<30){
  timer = millis();
  LightLevel = analogRead(sensorPin);
Serial.println("Nacht ");
}

// if button is pressed
season_button();
time_closed = time_difference(timer  ,millis()); 
// Serial.print("Time: ");
// Serial.println(millis()); //prints time since programm started

// Read the position of the flex sensor (0 to 1023):
  flexPosition = analogRead(flexPin);
  Serial.print("Zeit geschlossen: ");
  Serial.print(ventilate_time_closed);
  Serial.print("/");
  Serial.println(time_closed);
  if(ventilate_time_closed < time_closed ){ //Is it time to ventilate?
  ventilate_process();
  }     
    
//  Serial.print("sensor: ");
//  Serial.println(flexPosition);
  
  delay(20);
}

unsigned long time_difference(unsigned long Start, unsigned long End){
    unsigned long difference; 
    difference = End - Start;
    return difference; 
  }

void ventilate_process(){
    while(flexPosition < 800){  // Is the window open?
      season_button();
      digitalWrite(13, HIGH);   // Turn on the red LED till the window is opened
      tone(speakerPin,440); // play an A for "aufmachen" 
      flexPosition = analogRead(flexPin); // read the position of the window
      Serial.print("Flexwert: ");
      Serial.println(flexPosition);
      }
   noTone(speakerPin);
   digitalWrite(13, LOW);   // Turn off the red LED 
   time_closed = 0;
   timer = millis(); // set the time to the current time
  
   while(ventilate_time_total > time_open){  //Was the window long enough open?
      season_button();
      Serial.print("Zeit geöffnet: ");
      Serial.print(ventilate_time_total);
      Serial.print("/");
      Serial.println(time_open);
      Serial.print("Flexwert: ");
      Serial.print(flexPosition);
      time_open = time_difference(timer, millis());   
   }
   time_open = 0;
   while(flexPosition > 800){  // Is the window closed?
          season_button();
          digitalWrite(12, HIGH);   // Turn on the yellow LED till the window is closed
          tone(speakerPin,262 ); // play a C for "close"  
          flexPosition = analogRead(flexPin); // read the position of the window
          Serial.print("Flexwert: ");
          Serial.println(flexPosition);
   }           
   noTone(speakerPin);
   digitalWrite(12, LOW);    // Turn off the yellow LED   
   timer = millis();    // set the new timer
  }  


void season_button(){ 
  button_state = digitalRead(button_pin);
  if(button_state == LOW){
     season +=1;
     Serial.println("knopf gedrückt");
     season = season%4; 
    season_set(season);
 }
}

void season_set(int x){ // where x is the account how often he butten was pressed 
     int display_time = 2000;
  switch (x) {
    case 0: // spring
      // green
    digitalWrite(RED_PIN, LOW);
    digitalWrite(GREEN_PIN, HIGH);
    digitalWrite(BLUE_PIN, LOW);
    ventilate_time_total =   600000;
    ventilate_time_closed = 8160000;   
    break;
  case 1: // summer
    // organe or yellow not sure
    analogWrite(RED_PIN, 255);
    analogWrite(GREEN_PIN, 165);
    analogWrite(BLUE_PIN, 0 );
    ventilate_time_total =   1800000;
    ventilate_time_closed = 13050000;   
    break;
  case 2: // autumn
    // red
    digitalWrite(RED_PIN, HIGH);
    digitalWrite(GREEN_PIN, LOW);
    digitalWrite(BLUE_PIN, LOW);
    ventilate_time_total =   900000;
    ventilate_time_closed = 8325000;   
    break;
  case 3: // winter
    // blue
    analogWrite(RED_PIN, 5);
    analogWrite(GREEN_PIN, 215);
    analogWrite(BLUE_PIN, 255);
    ventilate_time_total =   300000;
    ventilate_time_closed = 5520000;   
    break;
  }  
  delay(display_time);
  digitalWrite(RED_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
  digitalWrite(BLUE_PIN, LOW);
}
