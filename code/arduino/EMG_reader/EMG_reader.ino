/**********************************************************/
/* Demo program for:                                      */
/*    Board: SHIELD-EKG/EMG + Olimexino328                */
/*  Manufacture: OLIMEX                                   */
/*  COPYRIGHT (C) 2012                                    */
/*  Designed by:  Penko Todorov Bozhkov                   */
/*   Module Name:   Sketch                                */
/*   File   Name:   ShieldEkgEmgDemo.ino                  */
/*   Revision:  Rev.A                                     */
/*    -> Added is suppport for all Arduino boards.        */
/*       This code could be recompiled for all of them!   */
/*   Date: 19.12.2012                                     */
/*   Built with Arduino C/C++ Compiler, version: 1.0.3    */
/**********************************************************/
/**********************************************************
Purpose of this programme is to give you an easy way to
connect Olimexino328 to ElectricGuru(TM), see:
https://www.olimex.com/Products/EEG/OpenEEG/EEG-SMT/resources/ElecGuru40.zip
where you'll be able to observe yours own EKG or EMG signal.
It is based on:
***********************************************************
* ModularEEG firmware for one-way transmission, v0.5.4-p2
* Copyright (c) 2002-2003, Joerg Hansmann, Jim Peters, Andreas Robinson
* License: GNU General Public License (GPL) v2
***********************************************************
For proper communication packet format given below have to be supported:
///////////////////////////////////////////////
////////// Packet Format Version 2 ////////////
///////////////////////////////////////////////
// 17-byte packets are transmitted from Olimexino328 at 256Hz,
// using 1 start bit, 8 data bits, 1 stop bit, no parity, 57600 bits per second.

// Minimial transmission speed is 256Hz * sizeof(Olimexino328_packet) * 10 = 43520 bps.

struct Olimexino328_packet
{
  uint8_t	sync0;		// = 0xa5
  uint8_t	sync1;		// = 0x5a
  uint8_t	version;	// = 2 (packet version)
  uint8_t	count;		// packet counter. Increases by 1 each packet.
  uint16_t	data[6];	// 10-bit sample (= 0 - 1023) in big endian (Motorola) format.
  uint8_t	switches;	// State of PD5 to PD2, in bits 3 to 0.
};
*/
/**********************************************************/
#include <compat/deprecated.h>
#include <FlexiTimer2.h>
//http://www.arduino.cc/playground/Main/FlexiTimer2

// All definitions
#define NUMCHANNELS 6
#define HEADERLEN 4
#define PACKETLEN (NUMCHANNELS * 2 + HEADERLEN + 1)
#define SAMPFREQ 256                      // ADC sampling rate 256
#define TIMER2VAL (1024/(SAMPFREQ))       // Set 256Hz sampling frequency
#define LED1  13
#define CAL_SIG 9
#define CHANNEL_NUM 3

// Global constants and variables
volatile unsigned char counter = 0;	  //Additional divider used to generate CAL_SIG
volatile unsigned int ADC_Value = 0;	  //ADC current value

//~~~~~~~~~~
// Functions
//~~~~~~~~~~

/****************************************************/
/*  Function name: Toggle_LED1                      */
/*  Parameters                                      */
/*    Input   :  No	                            */
/*    Output  :  No                                 */
/*    Action: Switches-over LED1.                   */
/****************************************************/
void Toggle_LED1(void){

 if((digitalRead(LED1))==HIGH){ digitalWrite(LED1,LOW); }
 else{ digitalWrite(LED1,HIGH); }

}


/****************************************************/
/*  Function name: toggle_GAL_SIG                   */
/*  Parameters                                      */
/*    Input   :  No	                            */
/*    Output  :  No                                 */
/*    Action: Switches-over GAL_SIG.                */
/****************************************************/
void toggle_GAL_SIG(void){

 if(digitalRead(CAL_SIG) == HIGH){ digitalWrite(CAL_SIG, LOW); }
 else{ digitalWrite(CAL_SIG, HIGH); }

}


/****************************************************/
/*  Function name: setup                            */
/*  Parameters                                      */
/*    Input   :  No	                            */
/*    Output  :  No                                 */
/*    Action: Initializes all peripherals           */
/****************************************************/
void setup() {

 noInterrupts();  // Disable all interrupts before initialization

 // LED1
 pinMode(LED1, OUTPUT);  //Setup LED1 direction
 digitalWrite(LED1,LOW); //Setup LED1 state
 pinMode(CAL_SIG, OUTPUT);

 // Timer2
 // Timer2 is used to setup the analag channels sampling frequency and packet update.
 // Whenever interrupt occures, the current read packet is sent to the PC
 // In addition the CAL_SIG is generated as well, so Timer1 is not required in this case!
 FlexiTimer2::set(TIMER2VAL, Timer2_Overflow_ISR);
 FlexiTimer2::start();

 // Serial Port
 Serial.begin(57600);
 //Set speed to 57600 bps

 // MCU sleep mode = idle.
 //outb(MCUCR,(inp(MCUCR) | (1<<SE)) & (~(1<<SM0) | ~(1<<SM1) | ~(1<<SM2)));

 interrupts();  // Enable all interrupts after initialization has been completed
}

/****************************************************/
/*  Function name: Timer2_Overflow_ISR              */
/*  Parameters                                      */
/*    Input   :  No	                            */
/*    Output  :  No                                 */
/*    Action: Determines ADC sampling frequency.    */
/****************************************************/
void Timer2_Overflow_ISR()
{
  // Toggle LED1 with ADC sampling frequency /2
  Toggle_LED1();

  //Read the ADC inputs and print to serial
  Serial.print(millis() ); Serial.print(",");
  for (int nCh = 0; nCh <= (CHANNEL_NUM-1); ++nCh)
  {
	  Serial.print(analogRead(nCh) );
	  Serial.print( (nCh==(CHANNEL_NUM-1) )?"\n":"," );
  }

  // Generate the CAL_SIGnal
  counter++;		// increment the devider counter
  if(counter == 12){	// 250/12/2 = 10.4Hz ->Toggle frequency
    counter = 0;
    toggle_GAL_SIG();	// Generate CAL signal with frequ ~10Hz
  }
}


/****************************************************/
/*  Function name: loop                             */
/*  Parameters                                      */
/*    Input   :  No	                            */
/*    Output  :  No                                 */
/*    Action: Puts MCU into sleep mode.             */
/****************************************************/
void loop() {

 __asm__ __volatile__ ("sleep");

}
