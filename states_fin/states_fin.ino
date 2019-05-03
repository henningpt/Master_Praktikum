# include "pitches.h"


const int ltime  = 1500; // long time
const int stime  = 150; // short time
const int sstime = 5;   // super short time

const int button_port  =  4;
const int led_port     =  11;
const int speaker_port =  8;

const String off = "off";
const String on  = "on";


int state      =  0;
int temp_state = -1;

bool exit_flag = 0;

const unsigned long c_work_time = 20000;
unsigned long work_time = c_work_time;

unsigned long current_time = 0;

unsigned long temp_time = 0;


int pitch_seq[] = {
  NOTE_B6, NOTE_G6,
};


int pitch_holds[] = {
  4, 4
 };

int pitch_number = 2;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

  pinMode(led_port, OUTPUT);
  pinMode(speaker_port, OUTPUT);
  pinMode(button_port, INPUT_PULLUP);


  led_blink(3, stime, 1);
  delay(2 * stime );
  led_blink(3, stime, 1);
  delay(750);
  led(off);

  play_music(pitch_seq, pitch_holds, pitch_number, 1);

  Serial.print("state = 0");



}


void loop() {
  // put your main code here, to run repeatedly:


  switch (state){
    case 0: led(off);
            current_time = 0;
            temp_time    = 0;
            work_time    = c_work_time;
            change_state(1, 0);
            break;

    case 1: led(on);
            if(current_time==0){
              current_time = millis();
              ptime("Zeit nach erstmaligem setzen der Zeit", current_time);
            }else if(millis() - current_time > work_time){
              state = 3;
              exit_flag = 1;
              break;
            }else{
              ptime("Zeitdifferenz nach negativem Test: ", millis() - current_time);
              Serial.print("\n");
              ptime("work_time: ", work_time);
              Serial.print("\n");
              Serial.print(millis() - current_time > work_time);
            }
            change_state(2, 200);
            break;

    case 2: led_blink(1, 200, 0);
            ptime("State 2: ", 99);
            change_state(1, 0);
            break;

    case 3: led_blink(1, 100, 0);
            if(exit_flag){
              play_music(pitch_seq, pitch_holds, pitch_number, 2);
              exit_flag = 0;
            }
            change_state(0, 0);
            break;

    default: led_blink(8, 50, 1);
             delay(1000);
             state = 0;
             break;
  }
}


void play_music(int melody[], int holds[], int number, int reps){
  for (int iter = 0; iter < reps * number; iter++){
    tone(speaker_port, melody[iter], 2e3 / holds[iter]);

    int note_pause = 1e3 / holds[iter] * 1.30;
    delay(note_pause);
  }
}


void ptime(String text, unsigned long value){
  Serial.print("\n\n" + text);
  Serial.print("\n");
  Serial.print(value);
}



void change_state(int new_state, unsigned int pause){
  if(button_hold(stime)){
            if(new_state == 2 && state == 1){
              temp_time = millis();
            }
            led(on);
            led(off);
            delay(pause);
            led(on);
            delay(1000);
            led(off);
            delay(0.5 * stime);
            if(!button_hold(sstime)){
               led_blink(8, 45, 1);
               delay(1000);
               temp_state = state;
               state      = new_state;
            }else{
              if(button_hold(ltime)){
                state = -1;
              }
            }
            if(new_state = state && state==1 && temp_state == 2){
            work_time = work_time + millis() - temp_time;
            temp_state = -1;
            ptime("State 2 change wtime: ", work_time);
            }
          }
          // delay(stime);
}


void led(String ledstate){
  if(ledstate == "off"){
  digitalWrite(led_port, LOW);
  }else if(ledstate == "on"){
    digitalWrite(led_port, HIGH);
  }
}


void led_blink(unsigned int reps, unsigned int interval, bool led_was_on){
    for(int i = 0; i < reps; i++){
        delay(interval);
        led(on);
        delay(interval);
        led(off);      }
    if(led_was_on){
      led(on);
    }
}


bool get_bstate(){
  if(digitalRead(button_port) == HIGH){
    return(true);
  }
  else{
    return(false);
  }
}


bool button_hold(unsigned int duration){
  if (get_bstate()){
    delay(duration);
    if(get_bstate()){
      return(true);
    }else{
      return(false);
    }
  }else{
    return(false);
  }
}
