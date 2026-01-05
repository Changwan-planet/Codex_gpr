!> \file MD_FFT_IFFT.f90

MODULE MD_FFT_IFFT

IMPLICIT NONE

!PRIVATE
!PUBLIC :: fft



CONTAINS

  SUBROUTINE fft(xxreal, xximag, S, f_real, f_imag)
  USE p_determination
 
  IMPLICIT NONE

  !INPUT DATA & INITIALIZATION
  INTEGER, INTENT(IN) :: S         !THE NUMBER OF SAMPLE POINT
  !INTEGER, PARAMETER :: S = 4096        !THE NUMBER OF SAMPLE POINT

  INTEGER              :: N       !THE NUMBER OF SAMPLE POINT FOR DO LOOP   
  REAL*8, PARAMETER    :: pi=Acos(-1.0)
  INTEGER              :: K       !SAMPLE POINT

  INTEGER :: N2                   !THE SPACING BETWEEN DUAL NODES

  INTEGER :: NU                   !GAMMA  
  INTEGER :: L                    !GAMMA FOR DO LOOP (ARRAY NUMBER BEING CONSIDERED)                 
                    
  INTEGER :: P
  INTEGER :: P_INTEGER
  INTEGER :: NU1                  !THE NU1 IS THE RIGHT SHIFT REQUIRED WHEN DETERMINING THE VALUE OF P.
  INTEGER :: I                    !THIS COUNTER MONITORS THE NUMBER OF DUAL NODE PAIRS THAT HAVE BEEN CONSIDERED.                                         !THE COUNTER I IS THE CONTROL FOR DETERMINING WHEN THE PROGRAM MUST SKIP.
  INTEGER :: I_INTEGER
 
  REAL*8 :: ARG
  REAL*8, DIMENSION(0:S-1), INTENT(IN) :: xxreal
  REAL*8, DIMENSION(0:S-1), INTENT(IN) :: xximag
  !INTEGER, INTENT(IN) :: S         !THE NUMBER OF SAMPLE POINT

  REAL*8, DIMENSION(0:S-1) :: xreal
  REAL*8, DIMENSION(0:S-1) :: ximag
  REAL*8, DIMENSION(0:S-1) :: f_real
  REAL*8, DIMENSION(0:S-1) :: f_imag
  !REAL*8, DIMENSION(0:S-1), INTENT(OUT) :: f_real
  !REAL*8, DIMENSION(0:S-1), INTENT(OUT) :: f_imag


  REAL*8 :: TREAL_K
  REAL*8 :: TIMAG_K
  REAL*8 :: TREAL_I
  REAL*8 :: TIMAG_I

  REAL*8 :: CC
  REAL*8 :: SS

  REAL*8 :: T1 !TEMPORARY VALUE
  REAL*8 :: T3  !TEMPORARY VALUE
  INTEGER :: T4 !TEMPORARY VALUE
  INTEGER :: yy,t,  Z, zz,zzz

  INTEGER, DIMENSION(:), ALLOCATABLE :: K_BINARY
  INTEGER, DIMENSION(:), ALLOCATABLE :: K_BINARY_SCALED
  INTEGER, DIMENSION(:), ALLOCATABLE :: K_BINARY_SCALED_REVERSED
  INTEGER, DIMENSION(:), ALLOCATABLE :: I_BINARY
  INTEGER, DIMENSION(:), ALLOCATABLE :: I_BINARY_REVERSED

  N=S

  xreal = xxreal
  ximag = xximag

  T1 = 0
  T3 = 0
  T4 = 0
  P = 0
  P_INTEGER = 0
  I = 0
  I_INTEGER = 0
  K = 0
  N2 = 0
  NU = 0
  NU1 = 0
  CC = 0
  SS = 0
  ARG = 0
  L = 1



  DO 
    IF (N==1) EXIT
    N=N/2
    NU=NU+1
  END DO

  ALLOCATE(K_BINARY(NU))
  ALLOCATE(K_BINARY_SCALED(NU))
  ALLOCATE(K_BINARY_SCALED_REVERSED(NU))
  ALLOCATE(I_BINARY(NU))
  ALLOCATE(I_BINARY_REVERSED(NU))

  NU1 = NU - 1
  N2 = S / 2**(L)

  T4 = S-1   


  !THE LAST VALUE OF THE K! I NEED THIS VALUE TO GET THE DIGIT OF 
  !THE MOST LARGE BINARY VALUE IN THIS CALCUATION.

  150 IF(L<=NU) THEN
 
  140      I=1
  
  130      CALL INTEGER2BINARY(K,NU,K_BINARY)
           CALL SCALE2RIGHT(K_BINARY,NU1,K_BINARY_SCALED)
           CALL REVERSED_BINARY(K_BINARY_SCALED,K_BINARY_SCALED_REVERSED)  
           CALL BINARY2INTEGER(K_BINARY_SCALED_REVERSED,P_INTEGER)

           ARG = 2*pi*P_INTEGER/S
 
           CC = COS(ARG)
           SS = SIN(ARG)
        
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
  !        XREAL(K) = DEFINED ABOVE
  !        XIMAG(K) = 0.0
         
  !         X(K) = XREAL(K) + j XIMAG(K)
  !         X(K+N2) = XREAL(K+N2) + j XIMAG(K+N2)
  !         X(K+N2) * W^p
           
  !        ** [FOURIER TRANSFORM]
  !         =[(XREAL(K+N2) + j XIMAG(K+N2)) * (COS(ARG) - j SIN(ARG)]
                
  !         =[ XREAL(K+N2) * COS(ARG) + XIMAG(K+N2) *  SIN(ARG)]
  !          j [ XIMAG(K+N2) * COS(ARG) -  XREAL(K+N2) * SIN(ARG) ]  
  !
  !        ** [INVERSE FOURIER TRANSFORM!
  !         =[XREAL(K+N2) + j XIMAG(K+N2) * (COS(ARG) + j SIN(ARG)]
         
  !         =[ XREAL(K+N2) * COS(ARG) - XIMAG(K+N2) *  SIN(ARG)]
  !          j [ XIMAG(K+N2) * COS(ARG) +  XREAL(K+N 2) * SIN(ARG) ]  
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   
           TREAL_K = XREAL(K+N2) * CC + XIMAG(K+N2) * SS
           TIMAG_K = XIMAG(K+N2) * CC - XREAL(K+N2) * SS
 
           XREAL(K+N2) = XREAL(K) - TREAL_K
           XIMAG(K+N2) = XIMAG(K) - TIMAG_K
     
           XREAL(K) = XREAL(K) + TREAL_K
           XIMAG(K) = XIMAG(K) + TIMAG_K
                
           K=K+1   !GO TO THE NEXT NODE
         
            IF(I==N2) THEN
           
                K = K + N2
              
                IF (K < S-1) THEN
                    GO TO 140              
            
                ELSE 

                    L = L+1
                    N2 = N2/2
                    NU1 = NU1 -1
                    K = 0
                    GO TO 150  !GO TO THE NEXT ARRAY
              
                END IF   
                 
            ELSE

               I=I+1

               GO TO 130

            END IF    

      ELSE 
      
  200    CALL INTEGER2BINARY(K,NU,I_BINARY)           
         CALL REVERSED_BINARY(I_BINARY,I_BINARY_REVERSED)
         CALL BINARY2INTEGER (I_BINARY_REVERSED,I_INTEGER)
        
          IF (I_INTEGER < K) THEN
              GO TO 300         
          ELSE
           
              TREAL_I = XREAL(K)
              TIMAG_I = XIMAG(K)

              XREAL(K) = XREAL(I_INTEGER)
              XIMAG(K) = XIMAG(I_INTEGER)
              XREAL(I_INTEGER) = TREAL_I
              XIMAG(I_INTEGER) = TIMAG_I

              GO TO 300
                   
          END IF
            
  300      IF (K==(S-1)) THEN

  !     ++++++++
  !++++++output++++++
  !     ++++++++
             DO zzz = 0, S-1

                 f_real (zzz) = xreal(zzz)
                 f_imag (zzz) = ximag(zzz)

              END DO

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

         

           ELSE
 
               K =  K+1
               GO TO 200
           
           END IF

      END IF

  DEALLOCATE(K_BINARY)
  DEALLOCATE(K_BINARY_SCALED)
  DEALLOCATE(K_BINARY_SCALED_REVERSED)
  DEALLOCATE(I_BINARY)
  DEALLOCATE(I_BINARY_REVERSED)
 
  END SUBROUTINE fft


  SUBROUTINE ifft(xxreal, xximag, S, if_real, if_imag)
  USE p_determination
 
   IMPLICIT NONE

  !INPUT DATA & INITIALIZATION
  INTEGER, INTENT(IN) :: S         !THE NUMBER OF SAMPLE POINT
  !INTEGER, PARAMETER :: S = 4096        !THE NUMBER OF SAMPLE POINT



  INTEGER              :: N       !THE NUMBER OF SAMPLE POINT FOR DO LOOP   
  REAL*8, PARAMETER    :: pi=Acos(-1.0)
  INTEGER              :: K       !SAMPLE POINT

  INTEGER :: N2                   !THE SPACING BETWEEN DUAL NODES

  INTEGER :: NU                   !GAMMA  
  INTEGER :: L                    !GAMMA FOR DO LOOP (ARRAY NUMBER BEING CONSIDERED)                 
                    
  INTEGER :: P
  INTEGER :: P_INTEGER
  INTEGER :: NU1                  !THE NU1 IS THE RIGHT SHIFT REQUIRED WHEN DETERMINING THE VALUE OF P.
  INTEGER :: I                    !THIS COUNTER MONITORS THE NUMBER OF DUAL NODE PAIRS THAT HAVE BEEN CONSIDERED.                                         !THE COUNTER I IS THE CONTROL FOR DETERMINING WHEN THE PROGRAM MUST SKIP.
  INTEGER :: I_INTEGER
 
  REAL*8 :: ARG
  REAL*8, DIMENSION(0:S-1), INTENT(IN) :: xxreal
  REAL*8, DIMENSION(0:S-1), INTENT(IN) :: xximag
  !INTEGER, INTENT(IN) :: S         !THE NUMBER OF SAMPLE POINT

  REAL*8, DIMENSION(0:S-1) :: xreal
  REAL*8, DIMENSION(0:S-1) :: ximag
  REAL*8, DIMENSION(0:S-1) :: if_real
  REAL*8, DIMENSION(0:S-1) :: if_imag
  !REAL*8, DIMENSION(0:S-1), INTENT(OUT) :: f_real
  !REAL*8, DIMENSION(0:S-1), INTENT(OUT) :: f_imag


  REAL*8 :: TREAL_K
  REAL*8 :: TIMAG_K
  REAL*8 :: TREAL_I
  REAL*8 :: TIMAG_I

  REAL*8 :: CC
  REAL*8 :: SS

  REAL*8 :: T1 !TEMPORARY VALUE
  REAL*8 :: T3  !TEMPORARY VALUE
  INTEGER :: T4 !TEMPORARY VALUE
  INTEGER :: yy,t,  Z, zz,zzz
  REAL*8 :: f_sam_in

  INTEGER, DIMENSION(:), ALLOCATABLE :: K_BINARY
  INTEGER, DIMENSION(:), ALLOCATABLE :: K_BINARY_SCALED
  INTEGER, DIMENSION(:), ALLOCATABLE :: K_BINARY_SCALED_REVERSED
  INTEGER, DIMENSION(:), ALLOCATABLE :: I_BINARY
  INTEGER, DIMENSION(:), ALLOCATABLE :: I_BINARY_REVERSED

  N=S

  xreal = xxreal
  ximag = xximag

  T1 = 0
  T3 = 0
  T4 = 0
  P = 0
  P_INTEGER = 0
  I = 0
  I_INTEGER = 0
  K = 0
  N2 = 0
  NU = 0
  NU1 = 0
  CC = 0
  SS = 0
  ARG = 0
  L = 1
  !+++++++++++++++++++++THE KEY OF THE IFFT++++++++++++++++++++++++++
  f_sam_in = 1.0 / S     
  !S is the number of the sample and at the same time the freqneucny sampling interval frequency
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



  DO 
    IF (N==1) EXIT
    N=N/2
    NU=NU+1
  END DO

  ALLOCATE(K_BINARY(NU))
  ALLOCATE(K_BINARY_SCALED(NU))
  ALLOCATE(K_BINARY_SCALED_REVERSED(NU))
  ALLOCATE(I_BINARY(NU))
  ALLOCATE(I_BINARY_REVERSED(NU))

  NU1 = NU - 1
  N2 = S / 2**(L)

  T4 = S-1   


  !THE LAST VALUE OF THE K! I NEED THIS VALUE TO GET THE DIGIT OF 
  !THE MOST LARGE BINARY VALUE IN THIS CALCUATION.

  150 IF(L<=NU) THEN
 
  140      I=1
  
  130      CALL INTEGER2BINARY(K,NU,K_BINARY)
           CALL SCALE2RIGHT(K_BINARY,NU1,K_BINARY_SCALED)
           CALL REVERSED_BINARY(K_BINARY_SCALED,K_BINARY_SCALED_REVERSED)  
           CALL BINARY2INTEGER(K_BINARY_SCALED_REVERSED,P_INTEGER)

           ARG = 2*pi*P_INTEGER/S
 
           CC = COS(ARG)
           SS = SIN(ARG)
        
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
  !        XREAL(K) = DEFINED ABOVE
  !        XIMAG(K) = 0
         
  !         X(K) = XREAL(K) + j XIMAG(K)
  !         X(K+N2) = XREAL(K+N2) + j XIMAG(K+N2)
  !         X(K+N2) * W^p
           
  !        ** [FOURIER TRANSFORM]
  !         =[XREAL(K+N2) + j XIMAG(K+N2) * (COS(ARG) - j SIN(ARG)]
                
  !         =[ XREAL(K+N2) * COS(ARG) + XIMAG(K+N2) *  SIN(ARG)]
  !          j [ XIMAG(K+N2) * COS(ARG) -  XREAL(K+N 2) * SIN(ARG) ]  
  !
  !        ** [INVERSE FOURIER TRANSFORM!
  !         =[XREAL(K+N2) + j XIMAG(K+N2) * (COS(ARG) + j SIN(ARG)]
         
  !         =[ XREAL(K+N2) * COS(ARG) - XIMAG(K+N2) *  SIN(ARG)]
  !          j [ XIMAG(K+N2) * COS(ARG) +  XREAL(K+N 2) * SIN(ARG) ]  
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   
           TREAL_K = XREAL(K+N2) * CC - XIMAG(K+N2) * SS
           TIMAG_K = XIMAG(K+N2) * CC + XREAL(K+N2) * SS
 
           XREAL(K+N2) = XREAL(K) - TREAL_K
           XIMAG(K+N2) = XIMAG(K) - TIMAG_K
     
           XREAL(K) = XREAL(K) + TREAL_K
           XIMAG(K) = XIMAG(K) + TIMAG_K
                
           K=K+1   !GO TO THE NEXT NODE
         
            IF(I==N2) THEN
           
                K = K + N2
              
                IF (K < S-1) THEN
                    GO TO 140              
            
                ELSE 

                    L = L+1
                    N2 = N2/2
                    NU1 = NU1 -1
                    K = 0
                    GO TO 150  !GO TO THE NEXT ARRAY
              
                END IF   
                 
            ELSE

               I=I+1

               GO TO 130

            END IF    

      ELSE 
      
  200    CALL INTEGER2BINARY(K,NU,I_BINARY)           
         CALL REVERSED_BINARY(I_BINARY,I_BINARY_REVERSED)
         CALL BINARY2INTEGER (I_BINARY_REVERSED,I_INTEGER)
        
          IF (I_INTEGER < K) THEN
              GO TO 300         
          ELSE
           
              TREAL_I = XREAL(K)
              TIMAG_I = XIMAG(K)

              XREAL(K) = XREAL(I_INTEGER)
              XIMAG(K) = XIMAG(I_INTEGER)
              XREAL(I_INTEGER) = TREAL_I
              XIMAG(I_INTEGER) = TIMAG_I

              GO TO 300
                   
          END IF
            
  300      IF (K==(S-1)) THEN

!                  ++++++++++++++++++++++++++++++++++++++++++++
!             ++++++muliplication of frequency sample interval++++++
!                  ++++++++++++++++++++++++++++++++++++++++++++

               DO zzz= 0 , S-1
                      xreal(zzz) = f_sam_in * xreal(zzz)  
                      ximag(zzz) = f_sam_in * ximag(zzz)
              END DO

            

  !     ++++++++
  !++++++output++++++
  !     ++++++++
             DO zzz = 0, S-1

                 if_real (zzz) = xreal(zzz)
                 if_imag (zzz) = ximag(zzz)

              END DO

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

         

           ELSE
 
               K =  K+1
               GO TO 200
           
           END IF

      END IF

  DEALLOCATE(K_BINARY)
  DEALLOCATE(K_BINARY_SCALED)
  DEALLOCATE(K_BINARY_SCALED_REVERSED)
  DEALLOCATE(I_BINARY)
  DEALLOCATE(I_BINARY_REVERSED)
 
  END SUBROUTINE ifft

END MODULE 


