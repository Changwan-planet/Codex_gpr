!> \file MD_HILBERT.f90 

MODULE MD_FFT_HILBERT
IMPLICIT NONE


CONTAINS

  SUBROUTINE fft_hilbert(xreal,S,a_ifft_real)
!  SUBROUTINE fft_hilbert(xreal,S,a_ifft_real,a_ifft_imag)


  USE MD_FFT_IFFT
  
  IMPLICIT NONE

  INTEGER :: i!
  INTEGER, INTENT(IN) :: S                               !the number of sample point     
  REAL*8, PARAMETER ::  pi = ACOS(-1.0)


  REAL*8, DIMENSION(0:S-1), INTENT(IN):: xreal           !the real part of RAW gpr data
  REAL*8, DIMENSION(0:S-1) :: ximag           !the imaginary part of RAW gpr data


  REAL*8, DIMENSION(0:S-1) :: fft_real        !the real part of FFT
  REAL*8, DIMENSION(0:S-1) :: fft_imag        !the imaginary part FFT


  REAL*8, DIMENSION(0:S-1) :: ifft_real       !the real part of IFFT
  REAL*8, DIMENSION(0:S-1) :: ifft_imag       !the imaginary part IFFT

  COMPLEX*8, DIMENSION(0:S-1) :: a_signal     !the analytical signal


  REAL*8, DIMENSION(0:S-1) :: h_real          !the real part of the Hibert transformed signal
  REAL*8, DIMENSION(0:S-1) :: h_imag          !the imag part of the Hibert transtormed signal

  REAL*8, DIMENSION(0:S-1) :: a_magnitude
  REAL*8, DIMENSION(0:S-1) :: log_a_magnitude


  REAL*8, DIMENSION(0:S-1) :: a_ifft_real     !the real part of the IFFT, Hibert transformed signal
  REAL*8, DIMENSION(0:S-1) :: a_ifft_imag     !the imag part of the IFFT, Hibert transtormed signal

  COMPLEX*8, DIMENSION(0:S-1) :: phase_shift !Ideally, this filter has magnitude  1 at all frequencies and introduces a phase shift of $ -pi/2 at each positive frequency and $ +pi/2 at each negative frequency


  INTEGER :: t, f

!  OPEN (10, FILE = "/home/changwan/GPR/A_SCOPE_GPR.txt", STATUS="OLD", FORM="FORMATTED", ACTION="READ")
!  OPEN (20, FILE =  "/home/changwan/HILBERT/HILBERT_output.txt",  STATUS="REPLACE",  ACTION="WRITE")


!read the RAW gpr data
!READ(10,*) xreal
!  ximag = 0.0

!  xreal=xreal-SUM(xreal)/SIZE(xreal)

  !PRINT*, SUM(xreal)/SIZE(xreal)


!  DO i = 0,S-1
!     PRINT*, xreal(i)
!  END DO


  !make a simple signal
  !f = 1
  ximag = 0.0
  !DO t=0, S-1
  !   xreal(t)=COS(2*pi*f*t/(S-1))
  !END DO
  !DO i=0, S-1
  !   PRINT *, xreal(i)
  !END DO


!PRINT *, "0=",xreal(0)
!PRINT *, "1=",xreal(1)
!PRINT *, "2=",xreal(2)
!!PRINT *, "3=",xreal(3)
!PRINT *, "4=",xreal(4)
!PRINT *, "5=",xreal(4092)
!PRINT *, "4093=",xreal(4093)
!PRINT *, "4094=",xreal(4094)
!PRINT *, "4095=",xreal(4095)
!PRINT *, "4096=",xreal(4096)



  !fourier transform
  CALL fft(xreal, ximag, S, fft_real, fft_imag)


  !shifted the phase by pi/2
  !filtered out the negative-frequency components of a siganl

  DO i = 0, S-1
     IF (i <=(S/2-1) ) THEN
!=====Positive freuquency zone=====
!Wrong!  phase_shift(i) = cmplx(fft_real(i), fft_imag(i)) * cmplx (cos(+pi/2), sin(+pi/2))
         phase_shift(i) = cmplx(fft_real(i), fft_imag(i)) * cmplx (cos(-pi/2), sin(-pi/2))

     ELSE  
!=====Negative frequency zone=====
!Wrong!  phase_shift(i) = cmplx(fft_real(i), fft_imag(i)) * cmplx (cos(-pi/2), sin(-pi/2))
         phase_shift(i) = cmplx(fft_real(i), fft_imag(i)) * cmplx (cos(+pi/2), sin(+pi/2))

     END IF
  END DO

  h_real = REAL(phase_shift)
  h_imag = AIMAG(phase_shift)
 
  !inverse fourier transform
  CALL ifft(h_real, h_imag, S, a_ifft_real, a_ifft_imag)

!  a_signal = cmplx(xreal,a_ifft_real) 

!PRINT *, "cmplx(xreal,a_iift_real)"
!PRINT *, "0=",a_signal(0)
!PRINT *, "1=",a_signal(1)
!PRINT *, "2=",a_signal(2)
!PRINT *, "3=",a_signal(3)
!PRINT *, "4=",a_signal(4)
!PRINT *, "5=",a_signal(4092)
!PRINT *, "4093=",a_signal(4093)
!PRINT *, "4094=",a_signal(4094)
!PRINT *, "4095=",a_signal(4095)
!PRINT *, ""


!  DO i = 0, S-1
!     a_magnitude (i) = SQRT(xreal(i)**2 + a_ifft_real(i)**2)
!  END DO

!  DO i = 0, S-1
!     log_a_magnitude (i)= 10* log10(a_magnitude(i))
!  END DO

  !write
!  DO i= 0, S-1
   
  !  PRINT*, xreal(i), a_signal(i)
!    WRITE(20,*) xreal(i), a_ifft_real(i), log_a_magnitude(i)

!  END DO
  
  END SUBROUTINE fft_hilbert

END MODULE MD_FFT_HILBERT

